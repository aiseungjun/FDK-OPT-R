from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

import data as _base


@dataclass(frozen=True)
class OODSpec:
    kind: str
    x: float
    scatter_sigma: float = 7.0


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _dose_shift_cfg(noise_cfg: dict, dose_scale: float) -> dict:
    cfg = dict(noise_cfg)
    base = float(cfg.get("dose_fraction", 1.0))
    cfg["dose_fraction"] = max(1e-6, base * float(dose_scale))
    return cfg


def _apply_gain(x: torch.Tensor, gain: float) -> torch.Tensor:
    return _clamp01(x * float(gain))


def _apply_gamma_lut(x: torch.Tensor, gamma: float) -> torch.Tensor:
    g = float(gamma)
    x = _clamp01(x)
    return _clamp01(torch.pow(x, g))


def _gaussian_kernel1d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = int(max(1, math.ceil(4.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / torch.sum(k)


def _gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x

    orig_shape = x.shape
    if x.dim() == 2:
        x4 = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x4 = x.unsqueeze(0)
    elif x.dim() == 4:
        x4 = x
    elif x.dim() == 5:
        b, t, c, h, w = x.shape
        x4 = x.reshape(b * t, c, h, w)
    else:
        raise ValueError(f"Unsupported tensor shape for blur: {tuple(x.shape)}")

    device, dtype = x4.device, x4.dtype
    k1 = _gaussian_kernel1d(sigma, device, dtype)
    c = int(x4.size(1))

    kx = k1.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k1.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

    pad = k1.numel() // 2
    x4 = F.pad(x4, (pad, pad, 0, 0), mode="reflect")
    x4 = F.conv2d(x4, kx, groups=c)
    x4 = F.pad(x4, (0, 0, pad, pad), mode="reflect")
    x4 = F.conv2d(x4, ky, groups=c)

    if len(orig_shape) == 2:
        return x4[0, 0]
    if len(orig_shape) == 3:
        return x4[0]
    if len(orig_shape) == 4:
        return x4
    b, t, c, h, w = orig_shape
    return x4.reshape(b, t, c, h, w)


def _apply_scatter_veil(
    clean: torch.Tensor, alpha: float, sigma: float = 7.0
) -> torch.Tensor:
    a = float(alpha)
    if a <= 0:
        return clean
    blurred = _gaussian_blur2d(clean, sigma=sigma)
    return _clamp01((1.0 - a) * clean + a * blurred)


def apply_ood_to_clean_and_cfg(
    clean: torch.Tensor, noise_cfg: dict, spec: OODSpec
) -> tuple[torch.Tensor, dict]:
    kind = str(spec.kind).lower()
    x = float(spec.x)

    if kind == "dose_shift":
        return clean, _dose_shift_cfg(noise_cfg, x)
    if kind == "gain":
        return _apply_gain(clean, x), dict(noise_cfg)
    if kind == "gamma":
        return _apply_gamma_lut(clean, x), dict(noise_cfg)
    if kind == "scatter":
        return _apply_scatter_veil(
            clean, alpha=x, sigma=float(spec.scatter_sigma)
        ), dict(noise_cfg)
    raise ValueError(f"Unknown OOD kind: {spec.kind}")


class OODDenoiseDataset(_base.DenoiseDataset):
    def __init__(self, *args, ood_spec: OODSpec, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ood_spec = ood_spec
        if self.mode not in {"n2c", "pretrain"}:
            raise ValueError(
                "OODDenoiseDataset supports only mode='n2c' or 'pretrain'."
            )

    def __getitem__(self, idx: int):
        if self.burst_size == 1:
            ref = self.index[idx]
            clean = self._load_frame(ref, bbox=None)
            clean_4d = clean.unsqueeze(0)
            clean_ood_4d, cfg_ood = apply_ood_to_clean_and_cfg(
                clean_4d, self.noise_cfg, self.ood_spec
            )
            noisy = _base.corrupt(clean_ood_4d.clone(), cfg_ood)
            return noisy.squeeze(0), clean_ood_4d.squeeze(0)

        patient_id, center_idx, indices = self.windows[idx]
        refs = self.sequence_map[patient_id]
        center_ref = refs[center_idx]

        center = self._load_frame_raw(center_ref)
        center = _base._select_middle_channel(center)
        if center.dim() == 2:
            center = center.unsqueeze(0)
        center_hw: Tuple[int, int] = tuple(center.shape[-2:])
        bbox = _base.compute_black_border_bbox(center)

        def _load_frame_with_ref_size(ref) -> torch.Tensor:
            frame = self._load_frame_raw(ref)
            frame = _base._select_middle_channel(frame)
            if frame.dim() == 2:
                frame = frame.unsqueeze(0)
            if tuple(frame.shape[-2:]) != center_hw:
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=center_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            return frame

        def _load_stack(idx_list: Sequence[int]) -> torch.Tensor:
            frames = [
                _base.crop_to_bbox(_load_frame_with_ref_size(refs[i]), bbox)
                for i in idx_list
            ]
            return torch.stack(frames, dim=0)

        clean_stack = _load_stack(indices)

        if self.patch_size and self.split in {"training", "validation"}:
            _, crop_coords = _base._random_crop_with_coords(
                clean_stack[self.burst_target_pos], self.patch_size
            )
            clean_stack = torch.stack(
                [
                    _base._crop_with_coords(frame, self.patch_size, crop_coords)
                    for frame in clean_stack
                ],
                dim=0,
            )

        clean_stack_ood, cfg_ood = apply_ood_to_clean_and_cfg(
            clean_stack, self.noise_cfg, self.ood_spec
        )
        clean_center = clean_stack_ood[self.burst_target_pos]

        noisy_frames = []
        for frame in clean_stack_ood:
            noisy_frames.append(_base.corrupt(frame.unsqueeze(0), cfg_ood).squeeze(0))
        noisy_stack = torch.stack(noisy_frames, dim=0)
        return noisy_stack, clean_center
