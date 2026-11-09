import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

def rms_norm(x: torch.Tensor, eps: float=0.0001) -> torch.Tensor:
    dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

def power_norm(x: torch.Tensor, dim: int=1, alpha: float=2.0, eps: float=0.0001) -> torch.Tensor:
    x = torch.abs(x)
    x_alpha = x ** alpha
    return x_alpha / (x_alpha.sum(dim=dim, keepdim=True) + eps)

def corrcoef_pt(x: torch.Tensor, rowvar: bool=True, clip: bool=False) -> torch.Tensor:
    original_dtype = x.dtype
    x = x.to(torch.float32)
    if not rowvar:
        x = x.transpose(-1, -2)
    mean = x.mean(dim=-1, keepdim=True)
    x = x - mean
    std = x.std(dim=-1, unbiased=True, keepdim=True) + 0.0001
    x_norm = x / std
    corr = x_norm @ x_norm.transpose(-1, -2)
    if clip:
        corr = corr.clamp(-1, 1)
    return corr.to(original_dtype)

@torch.no_grad()
def compute_local_correlation(x: torch.Tensor, image_size: Optional[Tuple[int, int]]=None) -> torch.Tensor:
    (B, C, KK, HW) = x.shape
    if image_size is None:
        H = W = int(HW ** 0.5)
    else:
        (H, W) = image_size
    x = x.permute(0, 3, 2, 1)
    coef = corrcoef_pt(x, clip=False)[:, :, KK // 2, :]
    coef = coef.reshape(B, H, W, KK).permute(0, 3, 1, 2)
    return coef

class GlobalChannelAttention(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(GlobalChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_dim * 2, in_dim * 2, 1, padding=0, bias=False)
        self.attention = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, 1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(out_dim, out_dim, 1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_mean = torch.mean(cond, dim=(2, 3), keepdim=True)
        cond_std = torch.std(cond, dim=(2, 3), keepdim=True)
        global_stat = torch.cat([cond_mean, cond_std], dim=1)
        global_stat = self.conv(global_stat)
        global_stat = rms_norm(global_stat)
        y = self.attention(global_stat)
        return x * y

def same_padding(kernel_size: int, dilation: int=1) -> int:
    eff = (kernel_size - 1) * dilation + 1
    pad_total = eff - 1
    if pad_total % 2 != 0:
        raise ValueError('No symmetric SAME padding for these parameters.')
    return pad_total // 2

class DIDBlock(nn.Module):

    def __init__(self, kernel_size: int=3, num_channels: int=3, hidden_channels: int=64, unfold_dilation: int=2, power_alpha: float=2.0):
        super(DIDBlock, self).__init__()
        self.kernel_size = kernel_size
        self.unfold_dilation = unfold_dilation
        self.reflect_pad = nn.ReflectionPad2d(same_padding(kernel_size, dilation=unfold_dilation))
        self.power_alpha = power_alpha
        self.feature_extractor = nn.Sequential(nn.Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))
        self.kernel_predictor = nn.Conv2d(hidden_channels + self.kernel_size ** 2, kernel_size * kernel_size, kernel_size=3, padding=1, bias=False)
        self.gca = GlobalChannelAttention(num_channels, hidden_channels)
        self.last_kernels: Optional[torch.Tensor] = None
        self.prev_kernels: Optional[torch.Tensor] = None

    def forward(self, inp: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], use_dilation: bool=True, mix_alpha: Optional[float]=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        (x_in, resi, prev_kernels) = inp
        (B, C, H, W) = x_in.shape
        (padding, dilation) = (0, self.unfold_dilation) if use_dilation else (1, 1)
        src = self.reflect_pad(x_in) if use_dilation else x_in
        patches = F.unfold(src, kernel_size=self.kernel_size, padding=padding, dilation=dilation)
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, -1)
        local_corr = compute_local_correlation(patches, image_size=(H, W))
        x_normed = rms_norm(x_in)
        features = self.feature_extractor(x_normed)
        if resi is not None:
            features = self.gca(features, cond=resi)
        features = torch.cat([features, local_corr], dim=1)
        features = rms_norm(features)
        kernel_logits = self.kernel_predictor(features)
        kernels = kernel_logits.view(B, self.kernel_size * self.kernel_size, -1)
        kernels = power_norm(kernels, dim=1, alpha=self.power_alpha, eps=0.0001)
        self.prev_kernels = prev_kernels
        self.last_kernels = kernels.clone()
        kernels_weighted = kernels.unsqueeze(1)
        filtered = (patches * kernels_weighted).sum(dim=2).view(B, C, H, W)
        if self.training:
            mix_alpha = torch.rand((B, 1, 1, 1), device=x_in.device)
            out = mix_alpha * x_in + (1.0 - mix_alpha) * filtered
        elif mix_alpha is None:
            out = filtered
        else:
            out = mix_alpha * x_in + (1.0 - mix_alpha) * filtered
        if use_dilation:
            return (out, x_in - out, self.prev_kernels)
        else:
            return (out, x_in - out, self.last_kernels.clone())

class IDFNet(nn.Module):

    def __init__(self, num_iter: int=10, kernel_size: int=3, num_channels: int=3, hidden_channels: int=64, halt_threshold: float=0.015, **block_kwargs):
        super().__init__()
        self.num_iter = num_iter
        self.halt_threshold = halt_threshold
        self.block = DIDBlock(kernel_size, num_channels, hidden_channels, unfold_dilation=2, **block_kwargs)

    def forward(self, x: torch.Tensor, adaptive_iter: bool=False, max_iter: Optional[int]=None, alpha_schedule: Optional[List[float]]=None) -> torch.Tensor:
        output = (x, None, None)
        if max_iter is None:
            max_iter = self.num_iter
        for i in range(max_iter):
            mix_alpha = alpha_schedule[i] if alpha_schedule is not None else None
            output = self.block(output, use_dilation=i % 2 == 0, mix_alpha=mix_alpha)
            if adaptive_iter and i % 2 == 1 and (i > 1):
                loss = self._compute_adaptive_loss(self.block)
                if loss < self.halt_threshold:
                    break
        return output[0]

    def _compute_adaptive_loss(self, block: 'DIDBlock') -> float:
        kernels = block.last_kernels
        prev_kernels = block.prev_kernels
        avg_kernel = kernels.mean(dim=-1)
        prev_avg_kernel = prev_kernels.mean(dim=-1)
        center_idx = block.kernel_size * block.kernel_size // 2
        loss = F.l1_loss(avg_kernel[:, center_idx], prev_avg_kernel[:, center_idx])
        return loss.item()
