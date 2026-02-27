import random
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


DEFAULT_LOW_DOSE_CFG = {
    "dose_fraction": 0.5,
    "peak": None,
    "sigma_read": None,
    "gamma": 2.2,
    "response": "gamma",
    "log_gain": 6.0,
    "quantize_bits": 8,
    "normalize_brightness": True,
    "corr_sigma": 0.0,
    "corr_alpha": 0.0,
    "corr_mix": 0.0,
    "stripe_amp": 0.0,
    "stripe_axis": "col",
    "mask_ratio": 0.02,
    "noise_level": 0.01,
}


def _require_peak_sigma_cfg(cfg: dict) -> tuple[float, float]:
    peak = cfg.get("peak", None)
    sigma_read = cfg.get("sigma_read", None)
    if peak is None or sigma_read is None:
        raise ValueError(
            "DEFAULT_LOW_DOSE_CFG['peak'] and ['sigma_read'] are not set. "
            "Run `python peak_sigma_estimate.py` in `paper_code/`, then copy "
            "`estimate_peak` and `estimate_sigma_read` into this file."
        )
    return float(peak), float(sigma_read)


R2R_INPUT_STRENGTH = 0.05
R2R_TARGET_STRENGTH = 0.005


BLACK_BORDER_THRESHOLD = 0.02
BLACK_BORDER_MIN_RUN = 3
BLACK_BORDER_MIN_SIZE = 32

CARDIAC_LEFT_DOMINANCE = "Left_Dominance"
CARDIAC_RIGHT_DOMINANCE = "Right_Dominance"
CARDIAC_DATA_ROOT = "cardio_data"
CARDIAC_ANON_ROOT = "anonymous_syntax_holdout"
CARDIAC_FRAME_PAT = re.compile(r"^(\d+)")
CARDIAC_SPLIT_SEED = 2026
CARDIAC_SPLIT_RATIOS = (0.7, 0.15, 0.15)
CARDIAC_RESIZE_HW = (256, 256)


@dataclass(frozen=True)
class FrameRef:
    dataset: str
    index: int


class OptLabelStore:
    def __init__(
        self,
        root: Path,
        file_map: dict[str, str],
        dataset_map: dict[str, str] | None = None,
    ):
        self.root = Path(root)
        self.file_map = dict(file_map)
        self.dataset_map = dict(dataset_map or {})
        self._h5: dict[str, h5py.File] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = {}
        return state

    def _open(self, key: str) -> h5py.File:
        rel = self.file_map.get(key)
        if rel is None:
            raise KeyError(f"Missing opt label file mapping for dataset key: {key}")
        path = self.root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing opt label file: {path}")
        return h5py.File(path, "r")

    def dataset(self, key: str):
        h5 = self._h5.get(key)
        if h5 is None:
            h5 = self._open(key)
            self._h5[key] = h5
        ds_name = self.dataset_map.get(key, "opt_label")
        return h5[ds_name]

    def files(self) -> dict[str, Path]:
        return {k: self.root / v for k, v in self.file_map.items()}

    def close(self):
        for h5 in self._h5.values():
            try:
                h5.close()
            except Exception:
                pass
        self._h5 = {}

    def __del__(self):
        self.close()


def _to_4d(x01: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if x01.dim() == 3:
        return x01.unsqueeze(0), 3
    if x01.dim() == 4:
        return x01, 4
    raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x01.shape)}")


def _resize_chw(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    if tuple(x.shape[-2:]) == tuple(size_hw):
        return x
    return F.interpolate(
        x.unsqueeze(0),
        size=size_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _apply_cardiac_resolution_rule(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    return _resize_chw(x, CARDIAC_RESIZE_HW)


def _count_leading_true(mask: torch.Tensor) -> int:
    if mask.dim() != 1:
        raise ValueError(f"Expected 1D mask, got shape {tuple(mask.shape)}")
    first_false = torch.where(~mask)[0]
    return int(mask.numel()) if first_false.numel() == 0 else int(first_false[0].item())


def _compute_black_border_bbox_once(
    x: torch.Tensor,
    *,
    threshold: float,
    min_run: int,
    min_size: int,
) -> Tuple[int, int, int, int]:
    if x.dim() == 4 and x.size(0) == 1:
        x = x[0]
    if x.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")

    _c, h, w = x.shape
    if h <= 0 or w <= 0:
        return 0, h, 0, w

    lum = x.mean(dim=0)
    h_mid0 = int(h * 0.25)
    h_mid1 = int(h * 0.75)
    w_mid0 = int(w * 0.25)
    w_mid1 = int(w * 0.75)
    interior = lum[h_mid0:h_mid1, w_mid0:w_mid1]
    interior_mean = (
        float(interior.mean().item()) if interior.numel() else float(lum.mean().item())
    )
    interior_std = (
        float(interior.std(unbiased=False).item())
        if interior.numel()
        else float(lum.std(unbiased=False).item())
    )
    mean_thresh = max(
        float(threshold), min(0.4 * interior_mean, interior_mean - 0.75 * interior_std)
    )
    std_thresh = max(1e-4, 0.5 * interior_std)

    row_mean = lum.mean(dim=1)
    col_mean = lum.mean(dim=0)
    row_std = lum.std(dim=1, unbiased=False)
    col_std = lum.std(dim=0, unbiased=False)
    row_dark_frac = (lum <= mean_thresh).float().mean(dim=1)
    col_dark_frac = (lum <= mean_thresh).float().mean(dim=0)

    row_black = (
        (row_mean <= mean_thresh) & (row_std <= std_thresh) & (row_dark_frac >= 0.6)
    )
    col_black = (
        (col_mean <= mean_thresh) & (col_std <= std_thresh) & (col_dark_frac >= 0.6)
    )

    top = _count_leading_true(row_black)
    bottom = _count_leading_true(row_black.flip(0))
    left = _count_leading_true(col_black)
    right = _count_leading_true(col_black.flip(0))

    if top < int(min_run):
        top = 0
    if bottom < int(min_run):
        bottom = 0
    if left < int(min_run):
        left = 0
    if right < int(min_run):
        right = 0

    top_i = int(top)
    bottom_i = int(h - bottom)
    left_i = int(left)
    right_i = int(w - right)

    if bottom_i <= top_i or right_i <= left_i:
        return 0, h, 0, w
    if (bottom_i - top_i) < int(min_size) or (right_i - left_i) < int(min_size):
        return 0, h, 0, w
    return top_i, bottom_i, left_i, right_i


def compute_black_border_bbox(
    x: torch.Tensor,
    *,
    threshold: float = BLACK_BORDER_THRESHOLD,
    min_run: int = BLACK_BORDER_MIN_RUN,
    min_size: int = BLACK_BORDER_MIN_SIZE,
) -> Tuple[int, int, int, int]:
    if x.dim() == 4 and x.size(0) == 1:
        x = x[0]
    if x.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")

    _c, h, w = x.shape
    if h <= 0 or w <= 0:
        return 0, h, 0, w

    top_abs = 0
    bottom_abs = h
    left_abs = 0
    right_abs = w
    current = x

    for _ in range(3):
        t, b, l, r = _compute_black_border_bbox_once(
            current, threshold=threshold, min_run=min_run, min_size=min_size
        )
        if t == 0 and b == current.shape[1] and l == 0 and r == current.shape[2]:
            break

        top_abs += t
        left_abs += l
        bottom_abs = top_abs + (b - t)
        right_abs = left_abs + (r - l)
        current = x[:, top_abs:bottom_abs, left_abs:right_abs]

    if bottom_abs <= top_abs or right_abs <= left_abs:
        return 0, h, 0, w
    if (bottom_abs - top_abs) < int(min_size) or (right_abs - left_abs) < int(min_size):
        return 0, h, 0, w
    return top_abs, bottom_abs, left_abs, right_abs


def crop_to_bbox(x: torch.Tensor, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
    top, bottom, left, right = bbox
    if x.dim() < 3:
        raise ValueError(f"Expected at least 3D tensor, got shape {tuple(x.shape)}")
    h = x.size(-2)
    w = x.size(-1)
    if top <= 0 and bottom >= h and left <= 0 and right >= w:
        return x
    return x[..., top:bottom, left:right]


def apply_gamma_to_linear(x_disp: torch.Tensor, gamma: float) -> torch.Tensor:
    if float(gamma) <= 0:
        return x_disp
    x = torch.clamp(x_disp, 0.0, 1.0)
    return torch.pow(x, float(gamma))


def apply_gamma_to_display(x_lin: torch.Tensor, gamma: float) -> torch.Tensor:
    if float(gamma) <= 0:
        return x_lin
    x = torch.clamp(x_lin, 0.0, 1.0)
    inv = 1.0 / max(float(gamma), 1e-6)
    return torch.pow(x, inv)


def apply_response_to_linear(
    x_disp: torch.Tensor,
    gamma: float,
    *,
    response: str = "gamma",
    log_gain: float = 6.0,
) -> torch.Tensor:
    x = torch.clamp(x_disp, 0.0, 1.0)
    resp = str(response).lower()
    if resp == "gamma":
        if float(gamma) <= 0:
            return x
        return torch.pow(x, float(gamma))
    if resp == "log":
        g = float(log_gain)
        denom = math.expm1(g)
        denom = denom if abs(denom) > 1e-12 else 1.0
        return torch.expm1(g * x) / denom
    raise ValueError(f"Unsupported response: {response}")


def apply_response_to_display(
    x_lin: torch.Tensor,
    gamma: float,
    *,
    response: str = "gamma",
    log_gain: float = 6.0,
) -> torch.Tensor:
    x = torch.clamp(x_lin, 0.0, 1.0)
    resp = str(response).lower()
    if resp == "gamma":
        if float(gamma) <= 0:
            return x
        inv = 1.0 / max(float(gamma), 1e-6)
        return torch.pow(x, inv)
    if resp == "log":
        g = float(log_gain)
        g = g if abs(g) > 1e-12 else 1.0
        return torch.log1p(x * math.expm1(g)) / g
    raise ValueError(f"Unsupported response: {response}")


def quantize_to_bits(x: torch.Tensor, bits: int | None) -> torch.Tensor:
    if bits is None:
        return x
    b = int(bits)
    if b <= 0:
        return x
    levels = float((1 << b) - 1)
    x = torch.clamp(x, 0.0, 1.0)
    return torch.round(x * levels) / levels


def make_correlated_noise(
    shape: Sequence[int], sigma_corr: float, alpha: float, device: torch.device
) -> torch.Tensor:
    b, c, h, w = shape
    n = torch.randn((b, c, h, w), device=device)
    if sigma_corr <= 0 or alpha <= 0:
        return n

    nf = torch.fft.rfft2(n, norm="ortho")
    fy = torch.fft.fftfreq(h, d=1.0, device=device).view(h, 1)
    fx = torch.fft.rfftfreq(w, d=1.0, device=device).view(1, (w // 2) + 1)
    f = torch.sqrt(fx * fx + fy * fy)
    f = torch.clamp(f, min=1.0 / max(h, w))
    filt = 1.0 / (f ** float(alpha))
    filt = filt / filt.max()

    nf = nf * filt
    out = torch.fft.irfft2(nf, s=(h, w), norm="ortho")
    out = out / (out.std(unbiased=False) + 1e-8) * float(sigma_corr)
    return out


def add_stripe_pattern(
    x: torch.Tensor, amp: float, axis: str = "col", smooth: int = 17
) -> torch.Tensor:
    if amp <= 0:
        return x
    b, c, h, w = x.shape
    device = x.device

    if axis == "col":
        v = torch.randn((w,), device=device)
        if smooth > 1:
            k = torch.ones((smooth,), device=device) / smooth
            v = F.conv1d(v.view(1, 1, -1), k.view(1, 1, -1), padding=smooth // 2).view(
                -1
            )
        v = v / (v.std(unbiased=False) + 1e-8) * float(amp)
        stripe = v.view(1, 1, 1, w).expand(b, c, h, w)
    else:
        v = torch.randn((h,), device=device)
        if smooth > 1:
            k = torch.ones((smooth,), device=device) / smooth
            v = F.conv1d(v.view(1, 1, -1), k.view(1, 1, -1), padding=smooth // 2).view(
                -1
            )
        v = v / (v.std(unbiased=False) + 1e-8) * float(amp)
        stripe = v.view(1, 1, h, 1).expand(b, c, h, w)

    return x + stripe


def simulate_low_dose_postprocessed(
    x_disp: torch.Tensor,
    *,
    dose_fraction: float,
    peak: float,
    sigma_read: float,
    gamma: float,
    corr_sigma: float,
    corr_alpha: float,
    corr_mix: float,
    stripe_amp: float,
    stripe_axis: str,
    clamp01: bool = True,
    response: str = "gamma",
    log_gain: float = 6.0,
    quantize_bits: int | None = None,
    normalize_brightness: bool = True,
) -> torch.Tensor:
    x4, orig_dim = _to_4d(x_disp.to(torch.float32))
    device = x4.device

    x_lin = apply_response_to_linear(x4, gamma, response=response, log_gain=log_gain)

    df = max(float(dose_fraction), 1e-6)
    pk = max(float(peak), 1e-6)
    x_counts = torch.clamp(x_lin, 0.0, 1.0) * pk * df

    y_counts = torch.poisson(x_counts)
    if float(sigma_read) > 0:
        y_counts = y_counts + torch.randn_like(y_counts) * float(sigma_read) * pk

    denom = pk * df if bool(normalize_brightness) else pk
    noisy_lin = y_counts / denom

    if float(corr_sigma) > 0 and float(corr_mix) > 0:
        corr = make_correlated_noise(
            noisy_lin.shape, float(corr_sigma), float(corr_alpha), device
        )
        noisy_lin = noisy_lin + float(corr_mix) * corr

    if float(stripe_amp) > 0:
        noisy_lin = add_stripe_pattern(noisy_lin, float(stripe_amp), str(stripe_axis))

    if clamp01:
        noisy_lin = torch.clamp(noisy_lin, 0.0, 1.0)

    noisy_disp = apply_response_to_display(
        noisy_lin, gamma, response=response, log_gain=log_gain
    )
    if clamp01:
        noisy_disp = torch.clamp(noisy_disp, 0.0, 1.0)
    noisy_disp = quantize_to_bits(noisy_disp, quantize_bits)

    if orig_dim == 3:
        return noisy_disp.squeeze(0)
    return noisy_disp


def simulate_low_dose_postprocessed_linear(
    x_disp: torch.Tensor,
    *,
    dose_fraction: float,
    peak: float,
    sigma_read: float,
    gamma: float,
    corr_sigma: float,
    corr_alpha: float,
    corr_mix: float,
    stripe_amp: float,
    stripe_axis: str,
    clamp01: bool = True,
    response: str = "gamma",
    log_gain: float = 6.0,
    normalize_brightness: bool = True,
) -> torch.Tensor:
    x4, orig_dim = _to_4d(x_disp.to(torch.float32))
    device = x4.device

    x_lin = apply_response_to_linear(x4, gamma, response=response, log_gain=log_gain)

    df = max(float(dose_fraction), 1e-6)
    pk = max(float(peak), 1e-6)
    x_counts = torch.clamp(x_lin, 0.0, 1.0) * pk * df

    y_counts = torch.poisson(x_counts)
    if float(sigma_read) > 0:
        y_counts = y_counts + torch.randn_like(y_counts) * float(sigma_read) * pk

    denom = pk * df if bool(normalize_brightness) else pk
    noisy_lin = y_counts / denom

    if float(corr_sigma) > 0 and float(corr_mix) > 0:
        corr = make_correlated_noise(
            noisy_lin.shape, float(corr_sigma), float(corr_alpha), device
        )
        noisy_lin = noisy_lin + float(corr_mix) * corr

    if float(stripe_amp) > 0:
        noisy_lin = add_stripe_pattern(noisy_lin, float(stripe_amp), str(stripe_axis))

    if clamp01:
        noisy_lin = torch.clamp(noisy_lin, 0.0, 1.0)

    if orig_dim == 3:
        return noisy_lin.squeeze(0)
    return noisy_lin


def corrupt(x_disp: torch.Tensor, cfg: dict) -> torch.Tensor:
    peak, sigma_read = _require_peak_sigma_cfg(cfg)
    return simulate_low_dose_postprocessed(
        x_disp,
        dose_fraction=cfg["dose_fraction"],
        peak=peak,
        sigma_read=sigma_read,
        gamma=cfg.get("gamma", 2.2),
        response=cfg.get("response", "gamma"),
        log_gain=cfg.get("log_gain", 6.0),
        quantize_bits=cfg.get("quantize_bits", None),
        normalize_brightness=cfg.get("normalize_brightness", True),
        corr_sigma=cfg.get("corr_sigma", 0.0),
        corr_alpha=cfg.get("corr_alpha", 0.0),
        corr_mix=cfg.get("corr_mix", 0.0),
        stripe_amp=cfg.get("stripe_amp", 0.0),
        stripe_axis=cfg.get("stripe_axis", "col"),
        clamp01=True,
    )


def estimate_poisson_gaussian_params(
    x_disp: torch.Tensor,
    *,
    gamma: float = 2.2,
    response: str = "gamma",
    log_gain: float = 6.0,
    patch_size: int = 8,
    stride: int | None = None,
    max_patches: int = 8192,
    flat_quantile: float = 0.3,
) -> tuple[float, float]:
    x4, _ = _to_4d(x_disp.to(torch.float32))

    if x4.shape[1] > 1:
        x4s = x4[:, :1]
    else:
        x4s = x4

    x_lin = apply_response_to_linear(x4s, gamma, response=response, log_gain=log_gain)
    x_lin = torch.clamp(x_lin, 0.0, 1.0)

    ps = int(patch_size)
    st = int(stride) if stride is not None else ps

    patches = x_lin.unfold(2, ps, st).unfold(3, ps, st)
    means = patches.mean(dim=(-1, -2))
    vars_ = patches.var(dim=(-1, -2), unbiased=False)

    means = means.reshape(-1)
    vars_ = vars_.reshape(-1)

    if means.numel() > max_patches:
        idx = torch.randperm(means.numel(), device=means.device)[:max_patches]
        means = means[idx]
        vars_ = vars_[idx]

    if 0.0 < float(flat_quantile) < 1.0 and means.numel() > 8:
        thr = torch.quantile(vars_, float(flat_quantile))
        sel = vars_ <= thr
        means = means[sel]
        vars_ = vars_[sel]

    if means.numel() < 8:
        return float("nan"), float("nan")

    A = torch.stack([means, torch.ones_like(means)], dim=1)

    sol = torch.linalg.lstsq(A, vars_.unsqueeze(1)).solution.squeeze(1)
    a = float(sol[0].item())
    b = float(sol[1].item())

    a = max(a, 1e-12)
    b = max(b, 0.0)
    peak_est = 1.0 / a
    sigma_read_est = math.sqrt(b)
    return float(peak_est), float(sigma_read_est)


def get_noise_level(cfg: dict) -> float:
    _require_peak_sigma_cfg(cfg)
    return float(cfg.get("noise_level", 0.05))


def _infer_dataset_kind(root: Path) -> str:
    if (root / CARDIAC_DATA_ROOT / CARDIAC_ANON_ROOT).exists() or (
        root / CARDIAC_ANON_ROOT
    ).exists():
        return "cardiac"
    raise FileNotFoundError(
        f"Could not infer cardiac dataset root from: {root}. "
        f"Expected '{CARDIAC_DATA_ROOT}/{CARDIAC_ANON_ROOT}' or '{CARDIAC_ANON_ROOT}'."
    )


def _select_middle_channel(frame: torch.Tensor) -> torch.Tensor:
    if frame.dim() == 2:
        return frame.unsqueeze(0)
    if frame.dim() == 3:
        if frame.shape[-1] in (1, 2, 3, 4):
            ch = min(1, frame.shape[-1] - 1)
            return frame[..., ch].unsqueeze(0)
        if frame.shape[0] in (1, 2, 3, 4):
            ch = min(1, frame.shape[0] - 1)
            return frame[ch, ...].unsqueeze(0)
    if frame.dim() > 3:
        return frame[0, ...].unsqueeze(0)
    return frame.unsqueeze(0)


def _cardiac_root(root: Path) -> Path:
    if (root / CARDIAC_DATA_ROOT / CARDIAC_ANON_ROOT).exists():
        return root / CARDIAC_DATA_ROOT / CARDIAC_ANON_ROOT
    if (root / CARDIAC_ANON_ROOT).exists():
        return root / CARDIAC_ANON_ROOT
    return root / CARDIAC_DATA_ROOT / CARDIAC_ANON_ROOT


def _list_cardiac_study_dirs(base: Path) -> list[Path]:
    studies: list[Path] = []
    for dom in (CARDIAC_LEFT_DOMINANCE, CARDIAC_RIGHT_DOMINANCE):
        dom_dir = base / dom
        if not dom_dir.exists():
            continue
        for study in sorted([p for p in dom_dir.iterdir() if p.is_dir()]):
            has_seq = False
            for view in ("LCA", "RCA"):
                if any((study / view).glob("*.npz")):
                    has_seq = True
                    break
            if has_seq:
                studies.append(study)
    return sorted(studies)


def _split_cardiac_studies(root: Path, split: str) -> list[Path]:
    studies = _list_cardiac_study_dirs(root)
    if not studies:
        raise ValueError(f"No cardiac studies found under {root}")
    if split not in {"training", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    n = len(studies)
    if n < 3:
        # Not enough studies to form a disjoint train/val/test split.
        return studies if split == "training" else []

    rng = random.Random(CARDIAC_SPLIT_SEED)
    ordered = list(studies)
    rng.shuffle(ordered)

    train_ratio, val_ratio, _test_ratio = CARDIAC_SPLIT_RATIOS
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

    if split == "training":
        return ordered[:train_end]
    if split == "validation":
        return ordered[train_end:val_end]
    return ordered[val_end:]


def _cardiac_frame_idx(path: Path, fallback: int) -> int:
    m = CARDIAC_FRAME_PAT.match(path.stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return fallback
    return fallback


def _list_cardiac_npz_files(base: Path, split: str) -> list[Path]:
    out: list[Path] = []
    for study in _split_cardiac_studies(base, split):
        for view in ("LCA", "RCA"):
            seq_dir = study / view
            if not seq_dir.exists():
                continue
            files = sorted(
                seq_dir.glob("*.npz"),
                key=lambda p: (_cardiac_frame_idx(p, 10**9), str(p)),
            )
            out.extend(files)
    return sorted(out)


def _count_cardiac_frames(npz_path: Path) -> int:
    with np.load(npz_path) as npz:
        arr = npz["pixel_array"]
    if arr.ndim == 2:
        return 1
    if arr.ndim == 3:
        return int(arr.shape[0])
    raise ValueError(f"Unsupported cardiac npz shape for {npz_path}: {arr.shape}")


def _load_cardiac_frame(npz_path: Path, frame_idx: int) -> np.ndarray:
    with np.load(npz_path) as npz:
        arr = npz["pixel_array"]
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        if frame_idx < 0 or frame_idx >= arr.shape[0]:
            raise IndexError(
                f"Frame index out of range: {frame_idx}/{arr.shape[0]} for {npz_path}"
            )
        return np.asarray(arr[frame_idx], dtype=np.float32)
        raise ValueError(f"Unsupported cardiac npz shape for {npz_path}: {arr.shape}")


def _group_cardiac_frames(
    root: Path,
    split: str,
) -> tuple[Dict[str, List[Tuple[int, FrameRef]]], Dict[tuple[str, int], Path]]:
    groups: Dict[str, List[Tuple[int, FrameRef]]] = {}
    path_map: Dict[tuple[str, int], Path] = {}

    for npz_path in _list_cardiac_npz_files(root, split):
        seq_key = str(npz_path.relative_to(root)).replace("/", "__").replace(".npz", "")
        frame_count = _count_cardiac_frames(npz_path)
        if frame_count <= 0:
            continue

        items: list[Tuple[int, FrameRef]] = []
        for local_idx in range(frame_count):
            ref = FrameRef(dataset=seq_key, index=local_idx)
            items.append((local_idx, ref))
            path_map[(seq_key, local_idx)] = npz_path
        groups[seq_key] = items

    return groups, path_map


def _load_frame_path(root_path: Path, ref: FrameRef) -> Path:
    _infer_dataset_kind(root_path)
    _, path_map = _group_cardiac_frames(_cardiac_root(root_path), "training")
    path = path_map.get((ref.dataset, ref.index))
    if path is not None:
        return path
    all_map: Dict[tuple[str, int], Path] = {}
    for split in ("training", "validation", "test"):
        _, m = _group_cardiac_frames(_cardiac_root(root_path), split)
        all_map.update(m)
    if (ref.dataset, ref.index) not in all_map:
        raise KeyError(f"Missing frame path for {ref.dataset}:{ref.index}")
    return all_map[(ref.dataset, ref.index)]


def load_frame_tensor(
    root: str | Path,
    ref: FrameRef,
    in_channels: int = 1,
    *,
    bbox: Tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    root_path = Path(root)
    _infer_dataset_kind(root_path)
    data_root = _cardiac_root(root_path)
    groups, path_map = _group_cardiac_frames(data_root, split="training")
    if ref.dataset not in groups:
        all_groups: Dict[str, List[Tuple[int, FrameRef]]] = {}
        all_map: Dict[tuple[str, int], Path] = {}
        for split in ("training", "validation", "test"):
            g, m = _group_cardiac_frames(data_root, split=split)
            all_groups.update(g)
            all_map.update(m)
        groups = all_groups
        path_map = all_map
    path = path_map.get((ref.dataset, ref.index))
    if path is None:
        raise KeyError(f"Missing frame path for {ref.dataset}:{ref.index}")
    arr = _load_cardiac_frame(path, ref.index)
    tensor = torch.from_numpy(arr).unsqueeze(0)

    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    tensor = _apply_cardiac_resolution_rule(tensor)
    if bbox is None:
        bbox = compute_black_border_bbox(tensor)
    return crop_to_bbox(tensor, bbox)


def get_frame_groups(
    root: str | Path = ".",
    split: str = "test",
    *,
    dataset_kind: str = "cardiac",
    pretrain: bool = False,
) -> Dict[int | str, List[Tuple[int, FrameRef]]]:
    root_path = Path(root)
    kind = dataset_kind.lower()
    if kind == "auto":
        kind = "cardiac"
    if kind != "cardiac":
        raise ValueError(
            f"Unsupported dataset kind for InterventionalCardiology: {dataset_kind}"
        )
    _infer_dataset_kind(root_path)
    groups, _path_map = _group_cardiac_frames(_cardiac_root(root_path), split)
    return groups


class DenoiseDataset(Dataset):
    def __init__(
        self,
        root: str | Path = ".",
        split: str = "training",
        patch_size: int | None = 256,
        mode: str = "n2c",
        in_channels: int = 1,
        noise_cfg: dict | None = None,
        *,
        dataset_kind: str = "cardiac",
        burst_size: int = 1,
        pretrain: bool = False,
        opt_label_root: str | Path | None = None,
        use_opt_risk: bool = False,
        opt_label_jitter: float = 0.0,
        burst_align: bool = False,
        burst_align_max_shift: float = 10.0,
        burst_causal: bool = False,
        burst_target: str = "center",
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.mode = mode
        self.in_channels = in_channels
        self.noise_cfg = (
            dict(DEFAULT_LOW_DOSE_CFG) if noise_cfg is None else dict(noise_cfg)
        )
        self.burst_size = int(burst_size)
        if self.burst_size not in {1, 5}:
            raise ValueError("burst_size must be 1 or 5")
        self.burst_causal = bool(burst_causal)
        self.burst_target = str(burst_target).lower()
        if self.burst_target not in {"center", "last"}:
            raise ValueError("burst_target must be one of: center, last")
        self.burst_target_pos = (
            self.burst_size // 2
            if self.burst_target == "center"
            else self.burst_size - 1
        )
        self.burst_align = bool(burst_align)
        self.burst_align_max_shift = float(burst_align_max_shift)

        self.dataset_kind = dataset_kind.lower()
        if self.dataset_kind == "auto":
            self.dataset_kind = "cardiac"
        if self.dataset_kind != "cardiac":
            raise ValueError(
                f"Unsupported dataset_kind for InterventionalCardiology: {self.dataset_kind}"
            )
        _infer_dataset_kind(self.root)

        self.pretrain = bool(pretrain)
        self.frame_path_map: Dict[tuple[str, int], Path] = {}

        self.opt_label_root = Path(opt_label_root) if opt_label_root else None
        self.opt_label_store = None
        self.opt_label_map = {}
        self.use_opt_risk = bool(use_opt_risk)
        self.opt_label_jitter = float(opt_label_jitter)
        if self.mode == "opt":
            if self.opt_label_root is None:
                raise ValueError("opt_label_root must be provided for opt mode")
            self._load_opt_label_index()

        data_root = _cardiac_root(self.root)
        groups, path_map = _group_cardiac_frames(data_root, self.split)
        self.frame_path_map = path_map
        if not groups:
            raise FileNotFoundError(
                f"No frames for split={split} under root={self.root}"
            )

        self.frame_groups = groups
        self.sequences: List[Tuple[int | str, List[FrameRef]]] = []
        for patient_id, items in groups.items():
            refs = [ref for _, ref in items]
            if refs:
                self.sequences.append((patient_id, refs))
        if not self.sequences:
            raise FileNotFoundError(f"All frame groups empty for split={split}")

        self.sequence_map = {pid: refs for pid, refs in self.sequences}

        if self.burst_size == 1:
            self.index: List[FrameRef] = [
                ref for _, refs in self.sequences for ref in refs
            ]
            if not self.index:
                raise FileNotFoundError("No frames indexed")
        else:
            self.windows: List[Tuple[int | str, int, List[int]]] = []
            for patient_id, refs in self.sequences:
                for center_idx in range(len(refs)):
                    indices = self._make_window_indices(len(refs), center_idx)
                    self.windows.append((patient_id, center_idx, indices))
            if not self.windows:
                raise FileNotFoundError("No windows indexed")

        if self.mode == "opt":
            self._validate_opt_index_coverage()
            self._validate_opt_shape_alignment(sample_count=8)

    def _load_opt_label_index(self) -> None:
        import json

        index_path = self.opt_label_root / "opt_label_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing opt label index: {index_path}")
        data = json.loads(index_path.read_text())
        self.opt_label_map = {
            key: {int(k): v for k, v in value.items()}
            for key, value in data["map"].items()
        }
        file_map = dict(data.get("files", {}))
        if not file_map:
            file_map = {k: f"{k}_opt_label.hdf5" for k in self.opt_label_map.keys()}
        dataset_map = dict(data.get("datasets", {}))
        self.opt_label_store = OptLabelStore(
            self.opt_label_root,
            file_map=file_map,
            dataset_map=dataset_map,
        )
        self._validate_opt_label_metadata()

    def _validate_opt_label_metadata(self) -> None:
        if self.opt_label_root is None:
            return
        if self.opt_label_store is None:
            return
        for _dataset_key, path in self.opt_label_store.files().items():
            if not path.exists():
                continue
            with h5py.File(path, "r") as h5:
                crop_attr = h5.attrs.get("black_border_crop", None)
                if crop_attr is None:
                    raise RuntimeError(
                        f"Legacy opt-label file detected without black-border metadata: {path}. "
                        "Regenerate opt labels with opt_flow_generater.py."
                    )
                if not bool(crop_attr):
                    raise RuntimeError(
                        f"Opt-label file has black_border_crop=False: {path}. "
                        "Regenerate opt labels without --no_black_border_crop."
                    )
                thr = h5.attrs.get("black_border_threshold", None)
                min_run = h5.attrs.get("black_border_min_run", None)
                min_size = h5.attrs.get("black_border_min_size", None)
                if thr is None or min_run is None or min_size is None:
                    raise RuntimeError(
                        f"Opt-label file missing crop-parameter metadata: {path}. "
                        "Regenerate opt labels with opt_flow_generater.py."
                    )
                if abs(float(thr) - float(BLACK_BORDER_THRESHOLD)) > 1e-8:
                    raise RuntimeError(
                        f"Opt-label crop threshold mismatch in {path}: "
                        f"file={float(thr)} current={float(BLACK_BORDER_THRESHOLD)}. "
                        "Regenerate opt labels."
                    )
                if int(min_run) != int(BLACK_BORDER_MIN_RUN) or int(min_size) != int(
                    BLACK_BORDER_MIN_SIZE
                ):
                    raise RuntimeError(
                        f"Opt-label crop parameter mismatch in {path}: "
                        f"file_min_run={int(min_run)} file_min_size={int(min_size)} "
                        f"current_min_run={int(BLACK_BORDER_MIN_RUN)} current_min_size={int(BLACK_BORDER_MIN_SIZE)}. "
                        "Regenerate opt labels."
                    )

    def __len__(self) -> int:
        return len(self.index) if self.burst_size == 1 else len(self.windows)

    def _iter_opt_label_refs(self) -> List[FrameRef]:
        if self.burst_size == 1:
            return list(self.index)
        refs: List[FrameRef] = []
        for patient_id, _center_idx, indices in self.windows:
            refs.append(self.sequence_map[patient_id][indices[self.burst_target_pos]])
        return refs

    def _validate_opt_index_coverage(self) -> None:
        missing: List[str] = []
        for ref in self._iter_opt_label_refs():
            idx_map = self.opt_label_map.get(ref.dataset)
            if idx_map is None or ref.index not in idx_map:
                missing.append(f"{ref.dataset}:{ref.index}")
                if len(missing) >= 10:
                    break
        if missing:
            sample = ", ".join(missing)
            raise RuntimeError(
                "Opt label index does not cover this split. "
                f"split={self.split}, first missing refs: {sample}. "
                "Regenerate opt labels with current cardiac training/validation split."
            )

    def _validate_opt_shape_alignment(self, sample_count: int = 8) -> None:
        check_n = min(int(sample_count), len(self))
        for idx in range(check_n):
            sample = self[idx]
            if self.use_opt_risk:
                inp, label, unc = sample
            else:
                inp, label = sample
                unc = None

            if self.burst_size == 1:
                inp_hw = tuple(inp.shape[-2:])
            else:
                inp_hw = tuple(inp[self.burst_target_pos].shape[-2:])
            label_hw = tuple(label.shape[-2:])
            if inp_hw != label_hw:
                raise RuntimeError(
                    "Opt input/label shape mismatch detected during dataset init. "
                    f"split={self.split}, idx={idx}, input_hw={inp_hw}, label_hw={label_hw}. "
                    "Regenerate opt labels with current preprocessing settings."
                )
            if unc is not None:
                unc_hw = tuple(unc.shape[-2:])
                if inp_hw != unc_hw:
                    raise RuntimeError(
                        "Opt input/risk shape mismatch detected during dataset init. "
                        f"split={self.split}, idx={idx}, input_hw={inp_hw}, unc_hw={unc_hw}. "
                        "Regenerate opt labels with current preprocessing settings."
                    )

    def _make_window_indices(self, seq_len: int, center_idx: int) -> List[int]:
        if self.burst_causal:
            left = max(1, self.burst_size - 1)
            return [
                min(max(center_idx - left + k, 0), seq_len - 1)
                for k in range(self.burst_size)
            ]
        radius = self.burst_size // 2
        return [
            min(max(i, 0), seq_len - 1)
            for i in range(center_idx - radius, center_idx + radius + 1)
        ]

    def _load_frame_raw(self, ref: FrameRef) -> torch.Tensor:
        path = self.frame_path_map.get((ref.dataset, ref.index))
        if path is None:
            raise KeyError(f"Missing frame path for {ref.dataset}:{ref.index}")
        arr = _load_cardiac_frame(path, ref.index).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = _apply_cardiac_resolution_rule(tensor)
        return tensor

    def _load_frame(
        self, ref: FrameRef, bbox: Tuple[int, int, int, int] | None = None
    ) -> torch.Tensor:
        tensor = self._load_frame_raw(ref)
        tensor = _select_middle_channel(tensor)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if bbox is None:
            bbox = compute_black_border_bbox(tensor)
        tensor = crop_to_bbox(tensor, bbox)
        if self.patch_size and self.split in {"training", "validation"}:
            tensor = _random_crop(tensor, self.patch_size)
        return tensor

    def _load_opt_label(
        self, ref: FrameRef, bbox: Tuple[int, int, int, int] | None = None
    ) -> torch.Tensor:
        if self.opt_label_store is None:
            raise RuntimeError("opt_label_store not initialized")
        dataset_key = ref.dataset
        idx_map = self.opt_label_map.get(dataset_key)
        if idx_map is None or ref.index not in idx_map:
            raise KeyError(f"Missing opt label for {dataset_key}:{ref.index}")
        mapped_idx = idx_map[ref.index]
        ds = self.opt_label_store.dataset(dataset_key)
        arr = ds[mapped_idx]
        tensor = torch.as_tensor(arr, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        tensor = _apply_cardiac_resolution_rule(tensor)
        if bbox is not None:
            tensor = crop_to_bbox(tensor, bbox)
        return tensor

    def _load_opt_risk(
        self, ref: FrameRef, bbox: Tuple[int, int, int, int] | None = None
    ) -> torch.Tensor:
        if self.opt_label_store is None:
            raise RuntimeError("opt_label_store not initialized")
        dataset_key = ref.dataset
        idx_map = self.opt_label_map.get(dataset_key)
        if idx_map is None or ref.index not in idx_map:
            raise KeyError(f"Missing opt label for {dataset_key}:{ref.index}")
        mapped_idx = idx_map[ref.index]
        ds = self.opt_label_store.dataset(dataset_key)
        if "opt_label_risk" not in ds.parent:
            raise KeyError("Missing opt_label_risk dataset in opt label file")
        arr = ds.parent["opt_label_risk"][mapped_idx]
        tensor = torch.as_tensor(arr, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        tensor = _apply_cardiac_resolution_rule(tensor)
        if bbox is not None:
            tensor = crop_to_bbox(tensor, bbox)
        return tensor

    def __getitem__(self, idx: int):
        if self.burst_size == 1:
            ref = self.index[idx]
            if self.mode == "opt":
                raw = self._load_frame_raw(ref)
                raw = _select_middle_channel(raw)
                if raw.dim() == 2:
                    raw = raw.unsqueeze(0)
                bbox = compute_black_border_bbox(raw)
                clean = crop_to_bbox(raw, bbox)
                crop_coords = None
                if self.patch_size and self.split in {"training", "validation"}:
                    clean, crop_coords = _random_crop_with_coords(
                        clean, self.patch_size
                    )
                clean_4d = clean.unsqueeze(0)
            else:
                clean = self._load_frame(ref, bbox=None)
                clean_4d = clean.unsqueeze(0)

            if self.mode in {"n2c", "pretrain"}:
                noisy = corrupt(clean_4d.clone(), self.noise_cfg)
                return noisy.squeeze(0), clean
            if self.mode in {"n2v", "n2self"}:
                noisy = corrupt(clean_4d.clone(), self.noise_cfg)
                mask = _mask_pixels(
                    noisy, mask_ratio=self.noise_cfg.get("mask_ratio", 0.02)
                )
                inp = _replace_masked(noisy, mask)
                return inp.squeeze(0), noisy.squeeze(0), mask.squeeze(0)

            if self.mode == "r2r":
                k = random.randint(0, 3)
                clean_aug = torch.rot90(clean_4d.clone(), k, dims=(2, 3))

                base = corrupt(clean_aug.clone(), self.noise_cfg)
                inp = _r2r_corrupt(base.clone(), R2R_INPUT_STRENGTH)
                target = _r2r_corrupt(base.clone(), R2R_TARGET_STRENGTH)
                return inp.squeeze(0), target.squeeze(0)

            if self.mode == "opt":
                noisy = corrupt(clean_4d.clone(), self.noise_cfg)
                label = self._load_opt_label(ref, bbox=bbox)
                if crop_coords is not None:
                    label = _crop_with_coords(label, self.patch_size, crop_coords)
                if self.use_opt_risk:
                    unc = self._load_opt_risk(ref, bbox=bbox)
                    if crop_coords is not None:
                        unc = _crop_with_coords(unc, self.patch_size, crop_coords)
                    return noisy.squeeze(0), label, unc
                return noisy.squeeze(0), label

            raise ValueError(f"Unsupported mode: {self.mode}")

        patient_id, center_idx, indices = self.windows[idx]
        refs = self.sequence_map[patient_id]
        center_ref = refs[center_idx]
        target_ref = refs[indices[self.burst_target_pos]]

        center = self._load_frame_raw(center_ref)
        center = _select_middle_channel(center)
        if center.dim() == 2:
            center = center.unsqueeze(0)
        center_hw = tuple(center.shape[-2:])
        bbox = compute_black_border_bbox(center)

        def _load_frame_with_ref_size(ref: FrameRef) -> torch.Tensor:
            frame = self._load_frame_raw(ref)
            frame = _select_middle_channel(frame)
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
                crop_to_bbox(_load_frame_with_ref_size(refs[i]), bbox) for i in idx_list
            ]
            if self.patch_size and self.split in {"training", "validation"}:
                _, crop_coords = _random_crop_with_coords(
                    frames[self.burst_target_pos], self.patch_size
                )
                frames = [
                    _crop_with_coords(frame, self.patch_size, crop_coords)
                    for frame in frames
                ]
            stack = torch.stack(frames, dim=0)
            return stack

        def _load_stack_opt(
            idx_list: Sequence[int],
        ) -> tuple[torch.Tensor, tuple[int, int] | None]:
            frames = []
            for i in idx_list:
                raw = _load_frame_with_ref_size(refs[i])
                raw = crop_to_bbox(raw, bbox)
                frames.append(raw)
            crop_coords = None
            if self.patch_size and self.split in {"training", "validation"}:
                _, crop_coords = _random_crop_with_coords(
                    frames[self.burst_target_pos], self.patch_size
                )
                frames = [
                    _crop_with_coords(frame, self.patch_size, crop_coords)
                    for frame in frames
                ]
            stack = torch.stack(frames, dim=0)
            return stack, crop_coords

        if self.mode == "opt":
            clean_stack, crop_coords = _load_stack_opt(indices)
        else:
            clean_stack = _load_stack(indices)
            crop_coords = None
        clean_center = clean_stack[self.burst_target_pos]

        def _corrupt_stack(stack: torch.Tensor) -> torch.Tensor:
            noisy_frames = []
            for frame in stack:
                noisy_frames.append(
                    corrupt(frame.unsqueeze(0), self.noise_cfg).squeeze(0)
                )
            return torch.stack(noisy_frames, dim=0)

        if self.mode in {"n2c", "pretrain"}:
            noisy_stack = _corrupt_stack(clean_stack)
            return noisy_stack.contiguous().clone(), clean_center.contiguous().clone()
        if self.mode in {"n2v", "n2self"}:
            noisy_stack = _corrupt_stack(clean_stack)
            center_noisy = noisy_stack[self.burst_target_pos].unsqueeze(0)
            mask = _mask_pixels(
                center_noisy, mask_ratio=self.noise_cfg.get("mask_ratio", 0.02)
            )
            masked_center = _replace_masked(center_noisy, mask).squeeze(0)
            noisy_stack = noisy_stack.clone()
            noisy_stack[self.burst_target_pos] = masked_center
            return (
                noisy_stack.contiguous().clone(),
                center_noisy.squeeze(0).contiguous().clone(),
                mask.squeeze(0).contiguous().clone(),
            )

        if self.mode == "r2r":
            k = random.randint(0, 3)
            clean_aug = torch.rot90(clean_stack, k, dims=(2, 3))
            inp_frames = []
            target_frames = []
            for frame in clean_aug:
                base = corrupt(frame.unsqueeze(0), self.noise_cfg).squeeze(0)
                inp_frames.append(
                    _r2r_corrupt(base.unsqueeze(0), R2R_INPUT_STRENGTH).squeeze(0)
                )
                target_frames.append(
                    _r2r_corrupt(base.unsqueeze(0), R2R_TARGET_STRENGTH).squeeze(0)
                )
            return (
                torch.stack(inp_frames, dim=0).contiguous().clone(),
                torch.stack(target_frames, dim=0).contiguous().clone(),
            )

        if self.mode == "opt":
            noisy_stack = _corrupt_stack(clean_stack)
            label = self._load_opt_label(target_ref, bbox=bbox)
            if crop_coords is not None:
                label = _crop_with_coords(label, self.patch_size, crop_coords)
            if self.use_opt_risk:
                unc = self._load_opt_risk(target_ref, bbox=bbox)
                if crop_coords is not None:
                    unc = _crop_with_coords(unc, self.patch_size, crop_coords)
                return (
                    noisy_stack.contiguous().clone(),
                    label.contiguous().clone(),
                    unc.contiguous().clone(),
                )
            return noisy_stack.contiguous().clone(), label.contiguous().clone()

        raise ValueError(f"Unsupported mode: {self.mode}")


def _mask_pixels(x: torch.Tensor, mask_ratio: float = 0.02) -> torch.Tensor:
    b, c, h, w = x.shape
    num = max(1, int(h * w * mask_ratio))
    mask = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.bool)
    for i in range(b):
        idx = torch.randperm(h * w, device=x.device)[:num]
        mask.view(b, 1, -1)[i, 0, idx] = True
    return mask


def _replace_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    out = x.clone()
    if mask.sum() == 0:
        return out
    num = int(mask.sum().item()) * c
    rand_vals = torch.rand((num,), device=x.device, dtype=x.dtype)
    out[mask.expand(-1, c, -1, -1)] = rand_vals
    return out


def _r2r_corrupt(x: torch.Tensor, strength: float) -> torch.Tensor:
    s = float(strength)
    if s <= 0.0:
        return torch.clamp(x, 0.0, 1.0)
    noise = torch.randn_like(x) * s
    return torch.clamp(x + noise, 0.0, 1.0)


def _random_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = x.shape
    if h == size and w == size:
        return x
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        mode = "reflect" if h > 1 and w > 1 else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        _, h, w = x.shape
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return x[:, top : top + size, left : left + size]


def _pad_to_size(x: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = x.shape
    if h >= size and w >= size:
        return x
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    mode = "reflect" if h > 1 and w > 1 else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)


def _random_crop_with_coords(
    x: torch.Tensor, size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    x = _pad_to_size(x, size)
    _, h, w = x.shape
    if h == size and w == size:
        return x, (0, 0)
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return x[:, top : top + size, left : left + size], (top, left)


def _crop_with_coords(
    x: torch.Tensor, size: int, coords: tuple[int, int]
) -> torch.Tensor:
    x = _pad_to_size(x, size)
    _, h, w = x.shape
    top, left = coords
    if h == size and w == size:
        return x
    top = min(max(int(top), 0), h - size)
    left = min(max(int(left), 0), w - size)
    return x[:, top : top + size, left : left + size]


def _cardiac_patient_key_from_sequence_key(seq_key: str) -> str:
    key = str(seq_key)
    if "__" not in key:
        return key
    parts = key.split("__")
    if len(parts) >= 3:
        return "__".join(parts[:-2])
    return parts[0]


class CardiacPatientFrameBudgetSampler(Sampler[int]):
    def __init__(
        self,
        dataset: "DenoiseDataset",
        frame_budget: int,
        *,
        seed: int | None = None,
    ) -> None:
        budget = int(frame_budget)
        if budget <= 0:
            raise ValueError(f"frame_budget must be > 0, got {frame_budget}")
        if len(dataset) <= 0:
            raise ValueError("dataset must contain at least one sample")
        self.dataset = dataset
        self.frame_budget = min(budget, len(dataset))
        self.seed = int(CARDIAC_SPLIT_SEED if seed is None else seed)
        self._epoch = 0
        self._patient_to_indices = self._build_patient_index(dataset)
        if not self._patient_to_indices:
            raise ValueError("Could not build cardiac patient index for sampling")
        self._patients = list(self._patient_to_indices.keys())

    @staticmethod
    def _build_patient_index(dataset: "DenoiseDataset") -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        burst_size = int(getattr(dataset, "burst_size", 1))
        if burst_size == 1:
            refs = list(getattr(dataset, "index", []))
            for sample_idx, ref in enumerate(refs):
                patient_key = _cardiac_patient_key_from_sequence_key(ref.dataset)
                out.setdefault(patient_key, []).append(sample_idx)
            for indices in out.values():
                indices.sort()
            return out

        windows = list(getattr(dataset, "windows", []))
        target_pos = int(getattr(dataset, "burst_target_pos", 0))
        sequence_map = dict(getattr(dataset, "sequence_map", {}))
        for sample_idx, (seq_key, _center_idx, indices) in enumerate(windows):
            refs = sequence_map.get(seq_key)
            if refs is None or not indices:
                continue
            pos = min(max(target_pos, 0), len(indices) - 1)
            ref_idx = int(indices[pos])
            if ref_idx < 0 or ref_idx >= len(refs):
                continue
            ref = refs[ref_idx]
            patient_key = _cardiac_patient_key_from_sequence_key(ref.dataset)
            out.setdefault(patient_key, []).append(sample_idx)
        for indices in out.values():
            indices.sort()
        return out

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        patients = list(self._patients)
        rng.shuffle(patients)

        selected: List[int] = []
        remaining = int(self.frame_budget)
        for patient_key in patients:
            if remaining <= 0:
                break
            patient_indices = self._patient_to_indices[patient_key]
            if len(patient_indices) <= remaining:
                selected.extend(patient_indices)
                remaining -= len(patient_indices)
            else:
                selected.extend(patient_indices[:remaining])
                remaining = 0
                break

        rng.shuffle(selected)
        self._epoch += 1
        return iter(selected)

    def __len__(self) -> int:
        return int(self.frame_budget)


def get_dataloaders(
    root: str | Path = ".",
    patch_size: int | None = 256,
    batch_size: int = 4,
    num_workers: int = 4,
    mode: str = "n2c",
    in_channels: int = 1,
    noise_cfg: dict | None = None,
    *,
    dataset_kind: str = "cardiac",
    burst_size: int = 1,
    pretrain: bool = False,
    opt_label_root: str | Path | None = None,
    use_opt_risk: bool = False,
    opt_label_jitter: float = 0.0,
    burst_align: bool = False,
    burst_align_max_shift: float = 10.0,
    burst_causal: bool = False,
    burst_target: str = "center",
    train_patient_frame_budget: int | None = None,
    train_patient_seed: int | None = None,
    val_patient_frame_budget: int | None = None,
    val_patient_seed: int | None = None,
):
    root_path = Path(root)
    kind = dataset_kind.lower()
    if kind == "auto":
        kind = "cardiac"
    if kind != "cardiac":
        raise ValueError(
            f"Unsupported dataset kind for InterventionalCardiology: {dataset_kind}"
        )
    _infer_dataset_kind(root_path)

    train_ds = DenoiseDataset(
        root_path,
        "training",
        patch_size,
        mode,
        in_channels,
        noise_cfg,
        dataset_kind=kind,
        burst_size=burst_size,
        pretrain=pretrain,
        opt_label_root=opt_label_root,
        use_opt_risk=use_opt_risk,
        opt_label_jitter=opt_label_jitter,
        burst_align=burst_align,
        burst_align_max_shift=burst_align_max_shift,
        burst_causal=burst_causal,
        burst_target=burst_target,
    )
    val_ds = DenoiseDataset(
        root_path,
        "validation",
        patch_size,
        mode,
        in_channels,
        noise_cfg,
        dataset_kind=kind,
        burst_size=burst_size,
        pretrain=pretrain,
        opt_label_root=opt_label_root,
        use_opt_risk=use_opt_risk,
        opt_label_jitter=opt_label_jitter,
        burst_align=burst_align,
        burst_align_max_shift=burst_align_max_shift,
        burst_causal=burst_causal,
        burst_target=burst_target,
    )
    train_sampler = None
    if (
        kind == "cardiac"
        and train_patient_frame_budget is not None
        and int(train_patient_frame_budget) > 0
    ):
        train_sampler = CardiacPatientFrameBudgetSampler(
            train_ds,
            int(train_patient_frame_budget),
            seed=train_patient_seed,
        )

    if train_sampler is None:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
        )
    val_sampler = None
    if (
        kind == "cardiac"
        and val_patient_frame_budget is not None
        and int(val_patient_frame_budget) > 0
    ):
        val_sampler = CardiacPatientFrameBudgetSampler(
            val_ds,
            int(val_patient_frame_budget),
            seed=val_patient_seed,
        )
    if val_sampler is None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
        )
    if mode == "opt":
        test_loader = None
    else:
        test_ds = DenoiseDataset(
            root_path,
            "test",
            None,
            mode,
            in_channels,
            noise_cfg,
            dataset_kind=kind,
            burst_size=burst_size,
            pretrain=pretrain,
            opt_label_root=opt_label_root,
            use_opt_risk=use_opt_risk,
            opt_label_jitter=opt_label_jitter,
            burst_align=burst_align,
            burst_align_max_shift=burst_align_max_shift,
            burst_causal=burst_causal,
            burst_target=burst_target,
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=num_workers
        )
    return train_loader, val_loader, test_loader
