import copy

import math

import weakref

import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES = ('centered_gray', 'highpass_3', 'highpass_9', 'highpass_33',
                 'difference_x', 'difference_y', 'column_profile', 'row_profile')


def check_noisy_input(x):
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 32:
        raise ValueError('Expected grayscale repeated B3HW with H,W >= 32')
    if x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
        raise ValueError('Only finite float32 input is supported')
    if not (torch.equal(x[:, 0], x[:, 1]) and torch.equal(x[:, 0], x[:, 2])):
        raise ValueError('The three channels must repeat one grayscale image')


def observed_noise_features(x):


    check_noisy_input(x)
    gray = x.mean(1, keepdim=True)
    centered = gray-gray.mean((-2, -1), keepdim=True)
    features = [centered]
    for kernel in (3, 9, 33):
        low = F.avg_pool2d(F.pad(gray, (kernel//2,)*4, mode='reflect'), kernel, 1)
        features.append(gray-low)
    dx = F.pad(torch.diff(gray, dim=-1), (0, 1, 0, 0), mode='replicate')
    dy = F.pad(torch.diff(gray, dim=-2), (0, 0, 0, 1), mode='replicate')
    features.extend((dx, dy, centered.mean(-2, keepdim=True).expand_as(gray),
                     centered.mean(-1, keepdim=True).expand_as(gray)))
    maps = torch.cat(features, dim=1)

    rms = (maps.square().mean((-2, -1))+1e-12).sqrt()
    mean_abs = maps.abs().mean((-2, -1))
    statistics = torch.cat((rms, mean_abs, torch.log(rms+1e-6)), dim=1)
    return maps, statistics


class NoiseEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(len(FEATURE_NAMES), 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(64+24, 96), nn.LayerNorm(96),
                                    nn.SiLU(), nn.Linear(96, self.embedding_dim))


        self.amplitude = nn.Linear(16, self.embedding_dim, bias=False)

    def forward(self, x):
        maps, statistics = observed_noise_features(x)
        spatial = self.spatial(maps)
        mean = spatial.mean((-2, -1))
        deviation = spatial-mean[:, :, None, None]
        std = (deviation.square().mean((-2, -1))+1e-12).sqrt()
        h = self.fusion(torch.cat((mean, std, statistics), dim=1))
        return h+self.amplitude(statistics[:, :16])


class BoundedFiLM(nn.Module):
    def __init__(self, embedding_dim=64, channels=32, gain_max=.05, bias_rms_fraction=.02):
        super().__init__()
        if (not math.isfinite(gain_max) or not math.isfinite(bias_rms_fraction)
                or gain_max <= 0 or bias_rms_fraction <= 0):
            raise ValueError('Positive finite feature-modulation bounds are required')
        self.channels = int(channels)
        self.gain_max, self.bias_rms_fraction = float(gain_max), float(bias_rms_fraction)
        self.mapping = nn.Linear(embedding_dim, 2*self.channels)
        nn.init.zeros_(self.mapping.weight)
        nn.init.zeros_(self.mapping.bias)

    def forward(self, features, embedding):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError('Unexpected restoration-backbone feature geometry')
        coefficients = self.mapping(embedding)
        gain, bias_units = coefficients.chunk(2, dim=1)
        gain = self.gain_max*gain.tanh()[:, :, None, None]
        with torch.no_grad():
            scale = features.detach().square().mean((1, 2, 3), keepdim=True).sqrt()
        bias = self.bias_rms_fraction*bias_units.tanh()[:, :, None, None]*scale
        return features*(1+gain)+bias, {'gain': gain, 'bias': bias, 'feature_rms': scale}


def residual_components(residual):


    if residual.ndim != 4 or not residual.is_floating_point():
        raise ValueError('Expected floating-point BCHW residual')
    dc = residual.mean((-2, -1), keepdim=True)
    row = residual.mean(-1, keepdim=True)
    column = residual.mean(-2, keepdim=True)
    axes = row+column-2*dc
    interaction = residual-row-column+dc
    return dc, axes, interaction


def compose_axis_repair(x, linear_residual, base_output, interaction_gain=None):


    if linear_residual.shape != x.shape or base_output.shape != x.shape:
        raise ValueError('Input, linear residual and v5 output shapes must match')
    _, learned_axes, _ = residual_components(linear_residual)
    _, _, fixed_interaction = residual_components(base_output-x)
    if interaction_gain is None:
        interaction_gain = x.new_ones((x.shape[0], 1, 1, 1))
    if (not isinstance(interaction_gain, torch.Tensor)
            or interaction_gain.shape != (x.shape[0], 1, 1, 1)
            or interaction_gain.device != x.device or interaction_gain.dtype != x.dtype
            or not bool(torch.isfinite(interaction_gain).all())
            or bool((interaction_gain < 0).any()) or bool((interaction_gain > 1).any())):
        raise ValueError('Interaction gain must be one finite [0,1] scalar per image, shaped B111')
    return x+learned_axes+interaction_gain*fixed_interaction, learned_axes, fixed_interaction


def mixed_difference(x):
    return torch.diff(torch.diff(x, dim=-1), dim=-2)


class AxisRepairDiagnostics(nn.Module):

    def __init__(self, base):
        super().__init__()

        self._base_reference = weakref.ref(base)
        self.domain = getattr(base, 'domain', 'unknown')
        self.eval()

    def config_dict(self):
        return {'kind': 'learned_axis_scalar_gated_v5_interaction', 'domain': self.domain,
                'axis_amplitude_cap': None, 'axis_low_frequency_fraction': None,
                'output_DC': 'preserve_input_mean_per_channel',
                'interaction_source': 'original_frozen_v5',
                'interaction_gain_range': [0., 1.],
                'interaction_gain_granularity': 'one scalar per image',
                'semantic_structure_guarantee': False,
                'constraint': 'Pinteraction(output-input)=g*Pinteraction(v5-input), 0<=g<=1; float tolerance only'}

    @torch.no_grad()
    def diagnostics(self, x, output, absolute_tolerance=2e-5, baseline=None,
                    rms_tolerance=2e-6):
        if (not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0
                or not math.isfinite(rms_tolerance) or rms_tolerance <= 0):
            raise ValueError('A positive finite absolute tolerance is required')
        check_noisy_input(x)
        if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
            raise ValueError('Diagnostic output must match the input geometry/device/dtype')
        finite = bool(torch.isfinite(output).all())
        if not finite:
            return {'passed': False, 'finite': False, 'domain': self.domain,
                    'DC_maxabs': None, 'interaction_projection_RMS': None,
                    'interaction_projection_peak': None}
        base = self._base_reference()
        if base is None:
            raise RuntimeError('The frozen v5 reference is no longer alive')
        reference = base(x) if baseline is None else baseline
        if reference.shape != x.shape or not bool(torch.isfinite(reference).all()):
            raise ValueError('Diagnostic baseline must be finite and match input geometry')
        dc_error = float((output-x).mean((-2, -1)).abs().max())
        _, _, source_interaction = residual_components(reference-x)
        _, _, actual_interaction = residual_components(output-x)
        dimensions=(1,2,3)
        energy=source_interaction.square().mean(dimensions)
        identifiable=energy > rms_tolerance**2
        fitted=(actual_interaction*source_interaction).mean(dimensions)/energy.clamp_min(1e-30)


        fitted=torch.where(energy > 1e-30,fitted,torch.zeros_like(fitted))
        projected=fitted.clamp(0.,1.)[:,None,None,None]*source_interaction
        error=actual_interaction-projected
        projection_rms=error.square().mean(dimensions).sqrt()
        projection_peak=error.abs().flatten(1).max(1).values
        range_ok=bool((~identifiable |
                       ((fitted >= -absolute_tolerance)&(fitted <= 1.+absolute_tolerance))).all())
        passed=bool(dc_error <= rms_tolerance and range_ok
                    and bool((projection_rms <= rms_tolerance).all())
                    and bool((projection_peak <= absolute_tolerance).all()))
        return {'passed': passed,
                'finite': finite, 'domain': self.domain, 'DC_maxabs': dc_error,
                'interaction_gain_estimated': [float(v) if bool(ok) else None for v,ok in zip(fitted,identifiable)],
                'interaction_gain_identifiable': [bool(ok) for ok in identifiable],
                'interaction_gain_in_range': range_ok,
                'interaction_projection_RMS': float(projection_rms.max()),
                'interaction_projection_peak': float(projection_peak.max()),
                'absolute_tolerance': absolute_tolerance, 'rms_tolerance': rms_tolerance,
                'residual_RMS': float((output-x).square().mean().sqrt()),
                'residual_peak': float((output-x).abs().max()),
                'RMS_and_peak_are_measurements_not_caps': True,
                'semantic_structure_guarantee': False}


class ContrastiveV5(nn.Module):
    def __init__(self, base, embedding_dim=64, projection_dim=32,
                 gain_max=.05, bias_rms_fraction=.02, candidate_guard=None,
                 interaction_gate_initial=.95):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        required = ('input_conv', 'feature_conv', 'scale_block1', 'spike_block1',
                    'spike_block2', 'scale_block2', 'refine_conv', 'output_conv', 'relu', 'tanh')
        if not hasattr(base, 'backbone') or not hasattr(base, 'project_residual'):
            raise TypeError('Expected the original v5 StructureGuard')
        if any(not hasattr(base.backbone, name) for name in required):
            raise TypeError('Frozen backbone does not expose the verified DEAL NE structure')
        if candidate_guard:
            raise ValueError('Legacy frequency/RMS candidate_guard is not used by axis repair')
        self.restorer = copy.deepcopy(base.backbone).requires_grad_(True)
        self.embedding_dim, self.projection_dim = int(embedding_dim), int(projection_dim)
        self.encoder = NoiseEncoder(self.embedding_dim)
        self.projection = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                        nn.SiLU(), nn.Linear(self.embedding_dim, self.projection_dim))
        self.noise_stats_head = nn.Linear(self.embedding_dim, 6)
        if (not math.isfinite(interaction_gate_initial)
                or not 0 < interaction_gate_initial < 1):
            raise ValueError('Initial interaction gate must lie strictly inside (0,1)')
        self.interaction_gate = nn.Linear(self.embedding_dim, 1)
        nn.init.zeros_(self.interaction_gate.weight)
        nn.init.constant_(self.interaction_gate.bias,
                          math.log(interaction_gate_initial/(1.-interaction_gate_initial)))
        self.film = nn.ModuleList([BoundedFiLM(self.embedding_dim, 32, gain_max, bias_rms_fraction)
                                   for _ in range(2)])
        self._axis_diagnostics = AxisRepairDiagnostics(self.base)
        self._restorer_initialized_from_loaded_base = False
        self._freeze_batchnorm()

    def _freeze_batchnorm(self):
        for module in self.restorer.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                module.requires_grad_(False)

    def initialize_restorer_from_base(self):


        if self._restorer_initialized_from_loaded_base:
            raise RuntimeError('Restorer already initialized from the loaded v5')
        self.restorer.load_state_dict(self.base.backbone.state_dict(), strict=True)
        self.restorer.requires_grad_(True)
        self._freeze_batchnorm()
        self._restorer_initialized_from_loaded_base = True
        return self

    @property
    def current_guard(self):
        return self._axis_diagnostics

    def train(self, mode=True):
        super().train(mode)
        self.base.requires_grad_(False).eval()
        self._freeze_batchnorm()
        self._axis_diagnostics.eval()
        return self

    def conditioned_backbone(self, x, h, bypass_film=False):


        b = self.restorer
        out = b.relu(b.input_conv(x))
        out = b.relu(b.feature_conv(out))
        first = second = None
        if not bypass_film:
            out, first = self.film[0](out, h)
        res = out
        out, res = b.scale_block1(out, res)
        out, res = b.spike_block1(out, res)
        out, res = b.spike_block2(out, res)
        out, res = b.scale_block2(out, res)
        if not bypass_film:
            out, second = self.film[1](out, h)
        out = b.relu(b.refine_conv(out))
        linear = b.output_conv(out)

        linear = linear.mean(1, keepdim=True).expand_as(x)
        return linear, first, second

    def unconditioned_prediction(self, x):


        check_noisy_input(x)
        h = self.encoder(x)
        gain = self.interaction_gain_from_embedding(h)
        linear, _, _ = self.conditioned_backbone(x, None, bypass_film=True)
        with torch.no_grad():
            baseline = self.base(x)
        return compose_axis_repair(x, linear, baseline, gain)[0]

    def interaction_gain_from_embedding(self, h):
        return torch.sigmoid(self.interaction_gate(h))[:, :, None, None]

    def forward_with_aux(self, x, embedding_override=None, bypass_conditioning=False):
        check_noisy_input(x)
        h = self.encoder(x)
        z = F.normalize(self.projection(h), dim=1, eps=1e-8)
        stats = self.noise_stats_head(h)
        used_h = h if embedding_override is None else embedding_override
        if (used_h.shape != h.shape or used_h.dtype != h.dtype or used_h.device != h.device
                or not bool(torch.isfinite(used_h).all())):
            raise ValueError('Embedding override must match the finite Bx64 latent tensor')
        if bypass_conditioning:
            with torch.no_grad():
                output = self.base(x)
            linear = output-x
            learned_axes = fixed_interaction = None
            gain = None
            first = second = None
        else:
            linear, first, second = self.conditioned_backbone(x, used_h)
            gain = self.interaction_gain_from_embedding(used_h)
            with torch.no_grad():
                baseline = self.base(x)
            output, learned_axes, fixed_interaction = compose_axis_repair(x, linear, baseline, gain)
        return output, {'embedding': h, 'conditioning_embedding': used_h,
                        'projection': z, 'noise_statistics': stats,
                        'raw_prediction': x+linear, 'raw_residual': linear,
                        'projected_residual': output-x,
                        'learned_axis_residual': learned_axes,
                        'frozen_v5_interaction_residual': fixed_interaction,
                        'axis_residual': learned_axes,
                        'frozen_interaction': fixed_interaction,
                        'interaction_gain': gain,
                        'gated_interaction': None if gain is None else gain*fixed_interaction,
                        'film_first': first, 'film_second': second,
                        'bypass_conditioning': bool(bypass_conditioning),
                        'candidate_guard_applied': False,
                        'axis_repair_applied': not bool(bypass_conditioning)}

    def forward(self, x):
        return self.forward_with_aux(x)[0]
