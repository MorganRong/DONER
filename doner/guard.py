from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class StructureGuard(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        domain: str,
        off_axis_fraction: float = .20,
        low_frequency_fraction: float = .10,
        low_frequency_cutoff: float = 1 / 32,
        scan_half_angle_degrees: float = 10.,
        scan_axis_fraction: float = .90,
        residual_rms_max: float = .12,
        residual_peak_max: float = .50,
    ):
        super().__init__()
        if domain not in ('stm', 'ir'):
            raise ValueError('domain must be stm or ir')
        self.backbone = backbone
        self.domain = domain
        values = dict(
            off_axis_fraction=off_axis_fraction,
            low_frequency_fraction=low_frequency_fraction,
            low_frequency_cutoff=low_frequency_cutoff,
            scan_half_angle_degrees=scan_half_angle_degrees,
            scan_axis_fraction=scan_axis_fraction,
            residual_rms_max=residual_rms_max,
            residual_peak_max=residual_peak_max,
        )
        if any(not math.isfinite(float(v)) for v in values.values()):
            raise ValueError('All guard parameters must be finite')
        if not 0 < low_frequency_fraction <= off_axis_fraction < scan_axis_fraction < 1:
            raise ValueError('Require 0 < low <= off-axis < scan fractions < 1')
        if not 0 < low_frequency_cutoff < .5 or not 0 < scan_half_angle_degrees < 45:
            raise ValueError('Invalid spectral region')
        if residual_rms_max <= 0 or residual_peak_max <= 0:
            raise ValueError('Residual budgets must be positive')
        for key, value in values.items():
            setattr(self, key, float(value))

    def config_dict(self) -> dict:
        return {
            'domain': self.domain,
            **{key: getattr(self, key) for key in (
                'off_axis_fraction', 'low_frequency_fraction',
                'low_frequency_cutoff', 'scan_half_angle_degrees',
                'scan_axis_fraction', 'residual_rms_max', 'residual_peak_max')},
            'fft_norm': 'ortho',
            'dc_change': 0.,
            'scan_axis': 'horizontal_rows' if self.domain == 'stm' else 'vertical_columns',
            'limits_apply_to': 'Input-relative complex Fourier correction, plus global per-image RMS/peak',
            'scope': 'No clean-structure classifier; whole-image frequency bounds do not certify local signal fidelity',
        }

    @staticmethod
    def _check_input(x: Tensor) -> None:
        if x.ndim != 4 or min(x.shape) <= 0 or min(x.shape[-2:]) < 2:
            raise ValueError('Expected nonempty BCHW input with spatial dimensions >= 2')
        if x.dtype != torch.float32:
            raise ValueError('StructureGuard requires float32 input; autocast/half precision is unsupported')

    def _frequency_grid(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_input(x)
        h, w = x.shape[-2:]
        fy = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype)[:, None]
        fx = torch.fft.rfftfreq(w, device=x.device, dtype=x.dtype)[None, :]
        return fy, fx

    def scan_frequency_mask(self, x: Tensor) -> Tensor:


        fy, fx = self._frequency_grid(x)
        axial, transverse = (fy.abs(), fx.abs()) if self.domain == 'stm' else (fx.abs(), fy.abs())
        return (transverse <= math.tan(math.radians(self.scan_half_angle_degrees)) * axial)[None, None]

    def frequency_fractions(self, x: Tensor) -> Tensor:

        fy, fx = self._frequency_grid(x)
        scan = self.scan_frequency_mask(x)
        fractions = torch.where(scan, x.new_tensor(self.scan_axis_fraction),
                                x.new_tensor(self.off_axis_fraction))
        low = (fy.square() + fx.square()) <= self.low_frequency_cutoff ** 2
        fractions = torch.where(low[None, None], self.low_frequency_fraction, fractions)


        dc = (fy == 0) & (fx == 0)
        return torch.where(dc[None, None], 0., fractions)

    def project_residual(self, x: Tensor, proposed_residual: Tensor) -> Tensor:


        self._check_input(x)
        if (proposed_residual.shape != x.shape or proposed_residual.dtype != x.dtype
                or proposed_residual.device != x.device):
            raise ValueError('Residual must share input shape, float32 dtype and device')
        spectrum = torch.fft.rfft2(x, norm='ortho')
        proposed = torch.fft.rfft2(proposed_residual, norm='ortho')
        radius = self.frequency_fractions(x) * spectrum.abs()


        scale = (radius / proposed.abs().clamp_min(1e-12)).clamp(max=1.)
        limited = torch.fft.irfft2(proposed * scale, s=x.shape[-2:], norm='ortho')

        squared_rms = limited.square().mean((1, 2, 3), keepdim=True)
        rms_scale = self.residual_rms_max / squared_rms.clamp_min(self.residual_rms_max ** 2).sqrt()
        peak_scale = self.residual_peak_max / limited.abs().amax((1, 2, 3), keepdim=True).clamp_min(self.residual_peak_max)
        return limited * torch.minimum(rms_scale, peak_scale)

    def forward_with_aux(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        self._check_input(x)
        raw = self.backbone(x)
        if raw.shape != x.shape or raw.dtype != x.dtype or raw.device != x.device:
            raise ValueError('Backbone must return a same-shape float32 image')
        proposed = raw - x
        limited = self.project_residual(x, proposed)
        output = x + limited
        return output, {'raw_prediction': raw, 'raw_residual': proposed,
                        'projected_residual': limited}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_aux(x)[0]

    @torch.no_grad()
    def diagnostics(self, x: Tensor, output: Tensor, absolute_tolerance: float = 2e-5) -> dict:

        self._check_input(x)
        if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
            raise ValueError('Diagnostic output must match input')
        if absolute_tolerance <= 0 or not math.isfinite(absolute_tolerance):
            raise ValueError('Invalid diagnostic tolerance')
        finite = bool(torch.isfinite(x).all() and torch.isfinite(output).all())
        if not finite:
            return {'passed': False, 'all_finite': False, 'absolute_tolerance': absolute_tolerance}
        correction = output - x
        spectrum = torch.fft.rfft2(x, norm='ortho')
        change = torch.fft.rfft2(correction, norm='ortho')
        excess = (change.abs() - self.frequency_fractions(x) * spectrum.abs()).clamp_min(0)
        rms = correction.square().mean((1, 2, 3)).sqrt()
        peak = correction.abs().amax((1, 2, 3))
        dc = correction.mean((-2, -1)).abs()
        result = {
            'all_finite': True,
            'absolute_tolerance': absolute_tolerance,
            'spectral_absolute_excess_max': float(excess.max()),
            'mean_change_absolute_max': float(dc.max()),
            'residual_rms_max_observed': float(rms.max()),
            'residual_peak_max_observed': float(peak.max()),
            'geometry_BCHW': list(x.shape),
            'parameters': self.config_dict(),
        }
        result['passed'] = bool(result['spectral_absolute_excess_max'] <= absolute_tolerance
            and result['mean_change_absolute_max'] <= absolute_tolerance
            and result['residual_rms_max_observed'] <= self.residual_rms_max + absolute_tolerance
            and result['residual_peak_max_observed'] <= self.residual_peak_max + absolute_tolerance)
        return result
