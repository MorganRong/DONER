import copy
import hashlib


from pathlib import Path


import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def state_sha(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def interaction(value):
    return (value - value.mean(-1, keepdim=True)
            - value.mean(-2, keepdim=True)
            + value.mean((-2, -1), keepdim=True))


def zero_dc(value):
    return value - value.mean((-2, -1), keepdim=True)


def highpass(value, size):
    smooth = F.avg_pool2d(F.pad(value, (size // 2,) * 4, mode='reflect'), size, 1)
    return value - smooth


def check_input(value):
    if (value.ndim != 4 or value.shape[1] != 1
            or min(value.shape[-2:]) < 32 or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())):
        raise ValueError('Expected finite float32 B1HW grayscale, H,W >=32')


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, padding_mode='reflect'),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode='reflect'))

    def forward(self, value):
        return value + .1 * self.body(value)


class FeatureFiLM(nn.Module):


    def __init__(self, width, embedding_dim=64):
        super().__init__()
        self.mapping = nn.Linear(embedding_dim, 2 * width)
        nn.init.zeros_(self.mapping.weight)
        nn.init.zeros_(self.mapping.bias)

    def forward(self, features, embedding):
        gain, bias = self.mapping(embedding).chunk(2, dim=1)
        return features * (1 + gain[:, :, None, None]) + bias[:, :, None, None]


class MultiscaleRestorer(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        widths = (32, 64, 128, 256)
        self.stem = nn.Conv2d(4, widths[0], 3, padding=1, padding_mode='reflect')
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(ResidualBlock(width), ResidualBlock(width)) for width in widths])
        self.encoder_film = nn.ModuleList([FeatureFiLM(width, embedding_dim) for width in widths])
        self.down = nn.ModuleList([
            nn.Conv2d(widths[i], widths[i + 1], 2, stride=2) for i in range(3)])
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(3)])
        self.fuse = nn.ModuleList([nn.Conv2d(widths[i] * 2, widths[i], 1) for i in range(3)])
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(ResidualBlock(width), ResidualBlock(width)) for width in widths[:-1]])
        self.decoder_film = nn.ModuleList([FeatureFiLM(width, embedding_dim) for width in widths[:-1]])
        self.head = nn.Conv2d(widths[0], 1, 3, padding=1, padding_mode='reflect')
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy, baseline, embedding):
        h, w = noisy.shape[-2:]
        features = torch.cat((noisy, baseline, highpass(noisy, 3), highpass(noisy, 9)), 1)


        pad_h, pad_w = (-h) % 8, (-w) % 8
        if pad_h or pad_w:
            features = F.pad(features, (0, pad_w, 0, pad_h), mode='reflect')
        value = self.stem(features)
        skips = []
        for level in range(4):
            value = checkpoint(self.encoder_blocks[level], value, use_reentrant=False) if self.training and torch.is_grad_enabled() else self.encoder_blocks[level](value)
            value = self.encoder_film[level](value, embedding)
            if level < 3:
                skips.append(value)
                value = self.down[level](value)
        for level in (2, 1, 0):
            value = self.up[level](value)
            value = self.fuse[level](torch.cat((value, skips[level]), 1))
            value = checkpoint(self.decoder_blocks[level], value, use_reentrant=False) if self.training and torch.is_grad_enabled() else self.decoder_blocks[level](value)
            value = self.decoder_film[level](value, embedding)
        return self.head(value)[..., :h, :w]


class ContrastiveV6R2(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.encoder = copy.deepcopy(teacher.encoder).requires_grad_(True)
        self.projection = copy.deepcopy(teacher.projection).requires_grad_(True)
        self.noise_stats_head = copy.deepcopy(teacher.noise_stats_head).requires_grad_(True)
        self.restorer = MultiscaleRestorer(teacher.embedding_dim)
        self.parent_state_sha256 = state_sha(self.teacher)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.requires_grad_(False).eval()
        return self

    @torch.no_grad()
    def baseline(self, noisy):
        check_input(noisy)
        return self.teacher(noisy.repeat(1, 3, 1, 1)).mean(1, keepdim=True)

    def forward_with_aux(self, noisy, embedding_override=None, baseline=None):
        check_input(noisy)
        if baseline is None:
            baseline = self.baseline(noisy)
        elif (baseline.shape != noisy.shape or baseline.dtype != noisy.dtype
              or baseline.device != noisy.device or baseline.requires_grad
              or not bool(torch.isfinite(baseline).all())):
            raise ValueError('Cached baseline must be a detached finite tensor matching input')
        embedding = self.encoder(noisy.repeat(1, 3, 1, 1))
        used_embedding = embedding if embedding_override is None else embedding_override
        if (used_embedding.shape != embedding.shape or used_embedding.dtype != embedding.dtype
                or used_embedding.device != embedding.device
                or not bool(torch.isfinite(used_embedding).all())):
            raise ValueError('Embedding override must match finite conditioning embedding')
        raw = self.restorer(noisy, baseline, used_embedding)
        correction = zero_dc(raw)
        output = baseline + correction
        return output, {
            'embedding': embedding,
            'conditioning_embedding': used_embedding,
            'projection': F.normalize(self.projection(embedding), dim=1, eps=1e-8),
            'noise_statistics': self.noise_stats_head(embedding),
            'baseline': baseline,
            'teacher_output': baseline,
            'raw_residual': raw,
            'correction': correction,
            'interaction_residual': interaction(raw),
            'row_residual': correction.mean(-1, keepdim=True),
            'column_residual': correction.mean(-2, keepdim=True),
            'axis_residual': (correction.mean(-1, keepdim=True)
                              + correction.mean(-2, keepdim=True)),
        }

    def forward_parts(self, noisy, baseline=None):
        output, aux = self.forward_with_aux(noisy, baseline=baseline)
        return output, aux['baseline'], aux['correction'], aux

    def forward(self, noisy):
        return self.forward_with_aux(noisy)[0]


from .backbone import NE
from .guard import StructureGuard
from .prior import ContrastiveV5

def read_state(path, domain=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if domain is not None and 'domain' in checkpoint and checkpoint['domain'] != domain:
        raise ValueError(f'Checkpoint domain {checkpoint["domain"]} does not match {domain}.')
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected a tensor state dictionary or a training checkpoint containing model')
    return state

def build_model(domain, prior=None):
    if domain not in ('stm','ir','confocal'):
        raise ValueError('Expected stm, ir or confocal')
    guard_domain = 'ir' if domain == 'ir' else 'stm'
    with torch.random.fork_rng(devices=[]):
        backbone = NE()
        with torch.no_grad():
            backbone.output_conv.conv2d.weight.mul_(0.01)
            backbone.output_conv.conv2d.bias.zero_()
    teacher = ContrastiveV5(StructureGuard(backbone, guard_domain))
    if prior is not None:
        state = read_state(prior)
        if any(name.startswith('teacher.') for name in state):
            state = {name.removeprefix('teacher.'): value for name,value in state.items() if name.startswith('teacher.')}
        teacher.load_state_dict(state, strict=True)
    return ContrastiveV6R2(teacher)

def load_model(domain, checkpoint, device='cpu'):
    net = build_model(domain)
    net.load_state_dict(read_state(checkpoint, domain=domain), strict=True)
    net.parent_state_sha256 = state_sha(net.teacher)
    return net.to(device).eval()
