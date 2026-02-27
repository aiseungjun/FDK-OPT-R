from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_int_hash(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _quantile_denom(x: np.ndarray, q: float = 0.99, eps: float = 1e-8) -> float:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 1.0
    v = float(np.quantile(arr, q))
    if not math.isfinite(v) or v <= 0:
        v = float(np.max(arr))
    return float(max(v, eps))


def _select_middle_channel(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        if frame.shape[-1] in (1, 2, 3, 4):
            return frame[..., 1 if frame.shape[-1] > 1 else 0]
        if frame.shape[0] in (1, 2, 3, 4):
            return frame[1 if frame.shape[0] > 1 else 0, ...]
    return frame.squeeze()


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


_HANNING_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _get_hanning_window(shape: Tuple[int, int]) -> np.ndarray:
    if shape not in _HANNING_CACHE:
        h, w = shape
        _HANNING_CACHE[shape] = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return _HANNING_CACHE[shape]


def _estimate_translation(
    src: np.ndarray, tgt: np.ndarray, max_shift: Optional[float]
) -> Tuple[float, float]:
    src_f = src.astype(np.float32)
    tgt_f = tgt.astype(np.float32)
    win = _get_hanning_window(src.shape)
    shift, _resp = cv2.phaseCorrelate(src_f, tgt_f, win)
    dx, dy = float(shift[0]), float(shift[1])
    if max_shift is not None:
        dx = float(np.clip(dx, -max_shift, max_shift))
        dy = float(np.clip(dy, -max_shift, max_shift))
    return dx, dy


def _warp_translation(src: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = src.shape[:2]
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    if src.ndim == 2:
        return cv2.warpAffine(
            src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    return cv2.warpAffine(
        src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def _warp_affine(src: np.ndarray, warp: np.ndarray) -> np.ndarray:
    h, w = src.shape[:2]
    return cv2.warpAffine(
        src,
        warp.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _ecc_rigid_warp(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    down: int,
    init_dx: float,
    init_dy: float,
    iters: int,
    eps: float,
) -> Tuple[np.ndarray, bool]:
    if down < 1:
        down = 1
    if down > 1:
        h, w = tgt.shape
        hs, ws = max(16, h // down), max(16, w // down)
        tgt_ds = cv2.resize(
            tgt.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA
        )
        src_ds = cv2.resize(
            src.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA
        )
    else:
        tgt_ds = tgt.astype(np.float32)
        src_ds = src.astype(np.float32)

    warp = np.array(
        [[1.0, 0.0, init_dx / down], [0.0, 1.0, init_dy / down]], dtype=np.float32
    )
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    try:
        _cc, warp_ds = cv2.findTransformECC(
            tgt_ds, src_ds, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 1
        )
        warp_full = warp_ds.copy()
        warp_full[0, 2] *= float(down)
        warp_full[1, 2] *= float(down)
        return warp_full.astype(np.float32), True
    except cv2.error:
        warp_full = np.array(
            [[1.0, 0.0, init_dx], [0.0, 1.0, init_dy]], dtype=np.float32
        )
        return warp_full, False


def _warp_with_flow(src: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = src.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def _warp_flow(flow: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    fx = cv2.remap(
        flow[..., 0],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    fy = cv2.remap(
        flow[..., 1],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.stack([fx, fy], axis=-1)


def _fb_error(flow_fw: np.ndarray, flow_bw: np.ndarray) -> np.ndarray:
    h, w = flow_fw.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_fw[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_fw[..., 1]).astype(np.float32)
    bw_warp = _warp_flow(flow_bw, map_x, map_y)
    return np.sqrt(
        (flow_fw[..., 0] + bw_warp[..., 0]) ** 2
        + (flow_fw[..., 1] + bw_warp[..., 1]) ** 2
    ).astype(np.float32)


def _build_flow(method: str):
    method = method.lower()
    if method == "tvl1" and hasattr(cv2, "optflow"):
        return cv2.optflow.DualTVL1OpticalFlow_create()

    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setFinestScale(1)
    flow.setPatchSize(8)
    flow.setPatchStride(4)
    flow.setGradientDescentIterations(25)
    return flow


def _anscombe(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)


ANS_COMB_MAX = 2.0 * math.sqrt(1.0 + 3.0 / 8.0)


def _prep_guide(x: np.ndarray, *, use_vst: bool, blur_sigma: float) -> np.ndarray:
    x = _clip01(x)
    if use_vst:
        x = _anscombe(x) / float(ANS_COMB_MAX)
    if blur_sigma > 0:
        x = cv2.GaussianBlur(
            x.astype(np.float32), ksize=(0, 0), sigmaX=float(blur_sigma)
        )
    return x.astype(np.float32)


def _grad_mag(x: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _edge_strength(x: np.ndarray) -> np.ndarray:
    mag = _grad_mag(x)
    denom = float(np.quantile(mag, 0.99)) + 1e-8
    return np.clip(mag / denom, 0.0, 1.0).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"NCC shape mismatch: {a.shape} vs {b.shape}")
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)) + 1e-8)
    return float(np.sum(a0 * b0) / denom)


def _robust_inlier_mask(
    stack: np.ndarray, trim_k: float, mad_eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(stack, axis=0)
    abs_diff = np.abs(stack - med[None, ...])
    mad = np.median(abs_diff, axis=0) + float(mad_eps)
    inlier = abs_diff <= (float(trim_k) * mad[None, ...] + float(mad_eps))
    return med.astype(np.float32), mad.astype(np.float32), inlier


def _weighted_mean(stack: np.ndarray, w: np.ndarray) -> np.ndarray:
    wsum = np.sum(w, axis=0)
    return (np.sum(w * stack, axis=0) / (wsum + 1e-8)).astype(np.float32)


def _weighted_var(stack: np.ndarray, w: np.ndarray, mean: np.ndarray) -> np.ndarray:
    wsum = np.sum(w, axis=0)
    return (np.sum(w * (stack - mean[None, ...]) ** 2, axis=0) / (wsum + 1e-8)).astype(
        np.float32
    )


def _effective_support(w_neighbors: np.ndarray, self_weight: float) -> np.ndarray:
    wsum_n = np.sum(w_neighbors, axis=0)
    return np.clip(wsum_n / (wsum_n + float(self_weight) + 1e-8), 0.0, 1.0).astype(
        np.float32
    )


def _foreground_mask(
    target: np.ndarray, warped_stack: np.ndarray, thresh: float, morph: int
) -> np.ndarray:
    if warped_stack.size == 0:
        return np.zeros_like(target, dtype=bool)
    med = np.median(warped_stack, axis=0).astype(np.float32)
    diff = np.abs(target.astype(np.float32) - med)
    mask = diff > float(thresh)
    if morph and int(morph) > 0:
        k = int(morph)
        kernel = np.ones((k, k), np.uint8)
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask = mask_u8 > 0
    return mask


@dataclass(frozen=True)
class FusionParams:
    radius: int = 4
    search_radius: int = 10
    max_neighbors: int = 8
    sim_down: int = 2
    sim_threshold: float = 0.10

    flow: str = "dis"
    use_vst: bool = True
    guide_blur_sigma: float = 1.2
    max_shift: float = 12.0
    use_ecc: bool = True
    ecc_down: int = 2
    ecc_iters: int = 30
    ecc_eps: float = 1e-5

    max_flow: float = 40.0
    fb_sigma: float = 4.0
    photo_sigma: float = 0.12
    grad_sigma: float = 0.10
    temporal_decay: float = 0.0

    self_weight: float = 0.7
    trim_k: float = 3.0
    mad_eps: float = 1e-4
    robust_k: float = 2.5

    fg_thresh: float = 0.08
    fg_morph: int = 3
    fg_keep: float = 0.25

    adaptive_hp_sigma: float = 1.6
    adaptive_ref_noise: float = 0.010
    max_update: float = 0.12

    detail_gain: float = 0.10
    detail_sigma: float = 1.2

    gate_support_pow: float = 1.2
    gate_mad_pow: float = 1.0
    gate_diff_pow: float = 1.0

    risk_var_scale: float = 0.004
    risk_mad_scale: float = 0.020
    risk_diff_scale: float = 0.050
    risk_edge_weight: float = 0.06

    stages: int = 2
    stage2_min_support: float = 0.20


def _compute_confidence(
    tgt_guide: np.ndarray,
    warped_guide: np.ndarray,
    flow_fw: np.ndarray,
    flow_bw: np.ndarray,
    *,
    max_flow: float,
    fb_sigma: float,
    photo_sigma: float,
    grad_sigma: float,
) -> np.ndarray:
    fb = _fb_error(flow_fw, flow_bw)
    conf_fb = np.exp(-fb / max(float(fb_sigma), 1e-6)).astype(np.float32)

    phot = np.abs(tgt_guide.astype(np.float32) - warped_guide.astype(np.float32))
    conf_ph = np.exp(-phot / max(float(photo_sigma), 1e-6)).astype(np.float32)

    g1 = _grad_mag(tgt_guide)
    g2 = _grad_mag(warped_guide)
    gd = np.abs(g1 - g2)
    conf_g = np.exp(-gd / max(float(grad_sigma), 1e-6)).astype(np.float32)

    flow_mag = np.sqrt(flow_fw[..., 0] ** 2 + flow_fw[..., 1] ** 2)
    conf_flow = (flow_mag <= float(max_flow)).astype(np.float32)

    return np.clip(conf_fb * conf_ph * conf_g * conf_flow, 0.0, 1.0).astype(np.float32)


def _select_neighbors(
    idx: int,
    total: int,
    *,
    get_obs_roi: Callable[[int], np.ndarray],
    tgt_guide_roi: np.ndarray,
    params: FusionParams,
    rng: np.random.Generator,
) -> List[Tuple[int, float, float, float]]:
    cand: List[int] = []
    sr = int(max(1, params.search_radius))
    for off in range(1, sr + 1):
        if idx - off >= 0:
            cand.append(idx - off)
        if idx + off < total:
            cand.append(idx + off)
    if not cand:
        return []

    down = int(max(1, params.sim_down))
    tgt_ds = tgt_guide_roi
    if down > 1:
        h, w = tgt_ds.shape
        tgt_ds = cv2.resize(
            tgt_ds,
            (max(16, w // down), max(16, h // down)),
            interpolation=cv2.INTER_AREA,
        )
    tgt_ds = tgt_ds.astype(np.float32)

    out: List[Tuple[int, float, float, float]] = []

    rng.shuffle(cand)
    for j in cand:
        src = get_obs_roi(int(j)).astype(np.float32)
        src_g = _prep_guide(
            src, use_vst=params.use_vst, blur_sigma=params.guide_blur_sigma
        )
        if down > 1:
            h, w = src_g.shape
            src_ds = cv2.resize(
                src_g,
                (max(16, w // down), max(16, h // down)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            src_ds = src_g

        dx, dy = _estimate_translation(
            src_ds,
            tgt_ds,
            params.max_shift / down if params.max_shift is not None else None,
        )
        src_ds_w = _warp_translation(src_ds, dx, dy)
        sim = _ncc(tgt_ds, src_ds_w)
        out.append((int(j), float(sim), float(dx * down), float(dy * down)))

    out.sort(key=lambda x: x[1], reverse=True)

    good = [t for t in out if t[1] >= float(params.sim_threshold)]
    return good[: int(max(0, params.max_neighbors))]


def _fuse_frame_once(
    idx: int,
    total: int,
    *,
    get_obs_roi: Callable[[int], np.ndarray],
    tgt_obs_roi: np.ndarray,
    ref_for_flow_roi: np.ndarray,
    params: FusionParams,
    flow_alg,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    tgt_obs_roi = _clip01(tgt_obs_roi)
    ref_for_flow_roi = _clip01(ref_for_flow_roi)

    tgt_guide = _prep_guide(
        ref_for_flow_roi, use_vst=params.use_vst, blur_sigma=params.guide_blur_sigma
    )
    tgt_u8 = _to_u8(tgt_guide)

    neigh = _select_neighbors(
        idx,
        total,
        get_obs_roi=get_obs_roi,
        tgt_guide_roi=tgt_guide,
        params=params,
        rng=rng,
    )
    if not neigh:
        h, w = tgt_obs_roi.shape
        return (
            tgt_obs_roi.astype(np.float32),
            np.ones((h, w), dtype=np.float32),
            {
                "support": np.zeros((h, w), dtype=np.float32),
                "mad_norm": np.ones((h, w), dtype=np.float32),
                "diff_norm": np.zeros((h, w), dtype=np.float32),
            },
        )

    warped_list: List[np.ndarray] = []
    conf_list: List[np.ndarray] = []

    for j, sim, dx, dy in neigh:
        src_obs = _clip01(get_obs_roi(int(j)))

        src_t = _warp_translation(src_obs, dx, dy)
        src_g = _prep_guide(
            src_t, use_vst=params.use_vst, blur_sigma=params.guide_blur_sigma
        )

        if params.use_ecc:
            warp_ecc, ok = _ecc_rigid_warp(
                src_g,
                tgt_guide,
                down=int(params.ecc_down),
                init_dx=0.0,
                init_dy=0.0,
                iters=int(params.ecc_iters),
                eps=float(params.ecc_eps),
            )
            if ok:
                src_t = _warp_affine(src_t, warp_ecc)
                src_g = _warp_affine(src_g, warp_ecc)

        src_u8 = _to_u8(src_g)

        flow_fw = flow_alg.calc(tgt_u8, src_u8, None)
        flow_bw = flow_alg.calc(src_u8, tgt_u8, None)

        warped = _warp_with_flow(src_t, flow_fw)
        warped_g = _warp_with_flow(src_g, flow_fw)

        conf = _compute_confidence(
            tgt_guide,
            warped_g,
            flow_fw,
            flow_bw,
            max_flow=params.max_flow,
            fb_sigma=params.fb_sigma,
            photo_sigma=params.photo_sigma,
            grad_sigma=params.grad_sigma,
        )

        sim_w = float(
            np.clip(
                (sim - params.sim_threshold) / max(1e-6, (1.0 - params.sim_threshold)),
                0.0,
                1.0,
            )
        )
        conf = conf * sim_w

        if params.temporal_decay and params.temporal_decay > 0:
            dist = abs(int(j) - int(idx))
            conf *= float(np.exp(-float(dist) / float(params.temporal_decay)))

        warped_list.append(_clip01(warped))
        conf_list.append(conf.astype(np.float32))

    warped_stack = np.stack(warped_list, axis=0)
    conf_stack = np.stack(conf_list, axis=0)

    fg_mask = _foreground_mask(
        tgt_obs_roi, warped_stack, params.fg_thresh, params.fg_morph
    )
    if float(params.fg_keep) < 1.0:
        conf_stack = np.where(
            fg_mask[None, ...], conf_stack * float(params.fg_keep), conf_stack
        )

    stack_safe = np.concatenate([warped_stack, tgt_obs_roi[None, ...]], axis=0)
    w_safe = np.concatenate(
        [
            conf_stack,
            np.full_like(
                tgt_obs_roi[None, ...], float(params.self_weight), dtype=np.float32
            ),
        ],
        axis=0,
    )

    _med, mad, inlier = _robust_inlier_mask(stack_safe, params.trim_k, params.mad_eps)
    w_safe = w_safe * inlier.astype(np.float32)
    safe1 = _weighted_mean(stack_safe, w_safe)
    resid = np.abs(stack_safe - safe1[None, ...])
    robust = np.exp(
        -((resid / (float(params.robust_k) * (mad[None, ...] + 1e-6))) ** 2)
    ).astype(np.float32)
    w_safe = w_safe * robust
    safe = _weighted_mean(stack_safe, w_safe)

    hp_sigma = max(float(params.adaptive_hp_sigma), 1e-6)
    lp = cv2.GaussianBlur(tgt_obs_roi.astype(np.float32), ksize=(0, 0), sigmaX=hp_sigma)
    hp = (tgt_obs_roi.astype(np.float32) - lp).astype(np.float32)

    edge = _edge_strength(tgt_obs_roi)
    mask_flat = edge < 0.25
    if np.any(mask_flat):
        med_hp = float(np.median(hp[mask_flat]))
        mad_hp = float(np.median(np.abs(hp[mask_flat] - med_hp))) + 1e-8
        noise_est = 1.4826 * mad_hp
    else:
        noise_est = float(np.std(hp))
    ref_noise = max(float(params.adaptive_ref_noise), 1e-8)
    noise_gate = float(np.clip((noise_est - ref_noise) / ref_noise, 0.0, 1.0))

    support = _effective_support(conf_stack, params.self_weight)
    conf_eff = np.clip(support + (1.0 - support) * noise_gate, 0.0, 1.0)

    delta = safe - tgt_obs_roi
    if params.max_update and float(params.max_update) > 0:
        delta = np.clip(delta, -float(params.max_update), float(params.max_update))
    safe = np.clip(tgt_obs_roi + conf_eff * delta, 0.0, 1.0).astype(np.float32)

    var_safe = _weighted_var(stack_safe, w_safe, safe)
    var_norm = np.clip(var_safe / (var_safe + float(params.risk_var_scale)), 0.0, 1.0)
    mad_norm = np.clip(mad / (mad + float(params.risk_mad_scale)), 0.0, 1.0)
    u_safe = (0.45 * (1.0 - support) + 0.30 * mad_norm + 0.25 * var_norm).astype(
        np.float32
    )

    w_ag = conf_stack.copy()
    _med2, mad2, inlier2 = _robust_inlier_mask(
        warped_stack, params.trim_k, params.mad_eps
    )
    w_ag = w_ag * inlier2.astype(np.float32)
    aggr = _weighted_mean(warped_stack, w_ag)

    if params.detail_gain and float(params.detail_gain) > 0:
        detail = tgt_obs_roi.astype(np.float32) - cv2.GaussianBlur(
            tgt_obs_roi.astype(np.float32),
            ksize=(0, 0),
            sigmaX=max(float(params.detail_sigma), 1e-6),
        )
        aggr = np.clip(aggr + float(params.detail_gain) * detail, 0.0, 1.0).astype(
            np.float32
        )

    var_ag = _weighted_var(warped_stack, w_ag, aggr)
    var_ag_norm = np.clip(var_ag / (var_ag + float(params.risk_var_scale)), 0.0, 1.0)
    mad2_norm = np.clip(mad2 / (mad2 + float(params.risk_mad_scale)), 0.0, 1.0)
    u_ag = (0.55 * (1.0 - support) + 0.25 * mad2_norm + 0.20 * var_ag_norm).astype(
        np.float32
    )

    diff = np.abs(aggr - safe).astype(np.float32)
    diff_norm = np.clip(diff / _quantile_denom(diff, 0.99), 0.0, 1.0)

    gate = np.power(np.clip(support, 0.0, 1.0), float(params.gate_support_pow))
    gate *= np.power(np.clip(1.0 - mad_norm, 0.0, 1.0), float(params.gate_mad_pow))
    gate *= np.power(np.clip(1.0 - diff_norm, 0.0, 1.0), float(params.gate_diff_pow))
    gate = np.clip(gate, 0.0, 1.0).astype(np.float32)

    gate = np.where(fg_mask, gate * 0.35, gate).astype(np.float32)

    label = (gate * aggr + (1.0 - gate) * safe).astype(np.float32)

    u_diff = np.clip(diff / (diff + float(params.risk_diff_scale)), 0.0, 1.0).astype(
        np.float32
    )
    unc = ((1.0 - gate) * u_safe + gate * u_ag + 0.25 * u_diff).astype(np.float32)

    unc = np.clip(
        unc
        + float(params.risk_edge_weight) * _edge_strength(tgt_obs_roi) * (1.0 - support),
        0.0,
        1.0,
    )

    debug = {
        "support": support.astype(np.float32),
        "mad_norm": mad_norm.astype(np.float32),
        "diff_norm": diff_norm.astype(np.float32),
        "gate": gate.astype(np.float32),
        "u_safe": u_safe.astype(np.float32),
    }
    return label.astype(np.float32), unc.astype(np.float32), debug


def run_sequence(
    *,
    seq_key: str,
    num_frames: int,
    get_clean_frame: Callable[[int], np.ndarray],
    data_mod: ModuleType,
    out_path: Path,
    params: FusionParams,
    max_frames: Optional[int],
    no_black_border_crop: bool,
    seed: int,
) -> Tuple[List[int], str]:
    total = int(num_frames)
    if max_frames is not None:
        total = min(total, int(max_frames))
    if total <= 0:
        raise ValueError(f"Empty sequence: {seq_key}")

    first = _select_middle_channel(get_clean_frame(0)).astype(np.float32)
    if first.max() > 1.5:
        first = first / 255.0
    h, w = first.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        try:
            with h5py.File(out_path, "r") as h5:
                if "opt_label" in h5 and "opt_label_risk" in h5:
                    if tuple(h5["opt_label"].shape) == (total, h, w):
                        return list(range(total)), out_path.name
        except Exception:
            pass

    noise_cfg = dict(getattr(data_mod, "DEFAULT_LOW_DOSE_CFG"))
    rng_global = np.random.default_rng(int(seed) + _stable_int_hash(seq_key))

    observed_cache: Dict[int, np.ndarray] = {}
    clean_cache: Dict[int, np.ndarray] = {}
    bbox_cache: Dict[int, Tuple[int, int, int, int]] = {}

    def _get_clean(i: int) -> np.ndarray:
        i = int(max(0, min(int(i), total - 1)))
        if i in clean_cache:
            return clean_cache[i]
        arr = _select_middle_channel(get_clean_frame(i)).astype(np.float32)
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = arr.astype(np.float32)
        clean_cache[i] = arr
        return arr

    def _corrupt(clean: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            if hasattr(data_mod, "simulate_low_dose_postprocessed"):
                noisy = data_mod.simulate_low_dose_postprocessed(
                    x,
                    dose_fraction=float(noise_cfg.get("dose_fraction", 0.5)),
                    peak=float(noise_cfg.get("peak", 120.0)),
                    sigma_read=float(noise_cfg.get("sigma_read", 0.01)),
                    gamma=float(noise_cfg.get("gamma", 2.0)),
                    corr_sigma=float(noise_cfg.get("corr_sigma", 0.0)),
                    corr_alpha=float(noise_cfg.get("corr_alpha", 1.0)),
                    corr_mix=float(noise_cfg.get("corr_mix", 0.0)),
                    stripe_amp=float(noise_cfg.get("stripe_amp", 0.0)),
                    stripe_axis=str(noise_cfg.get("stripe_axis", "col")),
                    clamp01=True,
                )
            else:
                noisy = data_mod.corrupt(x, noise_cfg)
        return noisy.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    def _get_observed(i: int) -> np.ndarray:
        i = int(max(0, min(int(i), total - 1)))
        if i in observed_cache:
            return observed_cache[i]
        obs = _corrupt(_get_clean(i))
        observed_cache[i] = obs
        return obs

    def _get_bbox(i: int) -> Tuple[int, int, int, int]:
        if no_black_border_crop or not hasattr(data_mod, "compute_black_border_bbox"):
            return (0, h, 0, w)
        i = int(max(0, min(int(i), total - 1)))
        if i in bbox_cache:
            return bbox_cache[i]
        x = torch.from_numpy(_get_clean(i)).float().unsqueeze(0)
        top, bottom, left, right = data_mod.compute_black_border_bbox(x)
        bbox = (
            max(0, min(int(top), h)),
            max(0, min(int(bottom), h)),
            max(0, min(int(left), w)),
            max(0, min(int(right), w)),
        )
        bbox_cache[i] = bbox
        return bbox

    flow_alg = _build_flow(params.flow)

    with h5py.File(out_path, "w") as out_h5:
        out_ds = out_h5.create_dataset(
            "opt_label",
            shape=(total, h, w),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        unc_ds = out_h5.create_dataset(
            "opt_label_risk",
            shape=(total, h, w),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )

        out_h5.attrs["dataset"] = str(seq_key)
        out_h5.attrs["label_strategy"] = "final_safeaggr_simselect_ecc_flow"
        out_h5.attrs["black_border_crop"] = bool(not no_black_border_crop)
        out_h5.attrs["black_border_threshold"] = float(
            getattr(data_mod, "BLACK_BORDER_THRESHOLD", 0.05)
        )
        out_h5.attrs["black_border_min_run"] = int(
            getattr(data_mod, "BLACK_BORDER_MIN_RUN", 16)
        )
        out_h5.attrs["black_border_min_size"] = int(
            getattr(data_mod, "BLACK_BORDER_MIN_SIZE", 16)
        )
        out_h5.attrs["stages"] = int(params.stages)
        out_h5.attrs["flow"] = str(params.flow)
        out_h5.attrs["radius"] = int(params.radius)
        out_h5.attrs["search_radius"] = int(params.search_radius)
        out_h5.attrs["max_neighbors"] = int(params.max_neighbors)

        for i in range(total):
            bbox = _get_bbox(i)
            top, bottom, left, right = bbox

            tgt_obs = _get_observed(i)
            tgt_obs_roi = tgt_obs[top:bottom, left:right].astype(np.float32)

            def get_obs_roi(j: int) -> np.ndarray:
                fr = _get_observed(int(j))
                return fr[top:bottom, left:right].astype(np.float32)

            label1_roi, unc1_roi, dbg1 = _fuse_frame_once(
                i,
                total,
                get_obs_roi=get_obs_roi,
                tgt_obs_roi=tgt_obs_roi,
                ref_for_flow_roi=tgt_obs_roi,
                params=params,
                flow_alg=flow_alg,
                rng=rng_global,
            )

            label_roi = label1_roi
            unc_roi = unc1_roi
            if int(params.stages) >= 2:
                support_mean = float(np.mean(dbg1["support"]))
                if support_mean >= float(params.stage2_min_support):
                    label2_roi, unc2_roi, _dbg2 = _fuse_frame_once(
                        i,
                        total,
                        get_obs_roi=get_obs_roi,
                        tgt_obs_roi=tgt_obs_roi,
                        ref_for_flow_roi=label1_roi,
                        params=params,
                        flow_alg=flow_alg,
                        rng=rng_global,
                    )
                    label_roi = label2_roi
                    unc_roi = unc2_roi

            label_full = np.zeros((h, w), dtype=np.float32)
            unc_full = np.ones((h, w), dtype=np.float32)
            label_full[top:bottom, left:right] = label_roi.astype(np.float32)
            unc_full[top:bottom, left:right] = unc_roi.astype(np.float32)

            out_ds[i] = label_full
            unc_ds[i] = unc_full

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"    [{i + 1:>5}/{total}] {seq_key}")

    return list(range(total)), out_path.name


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    repo: Path
    dataset_kind: str
    out_dir: str


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "weiss": DatasetConfig(
        "weiss", ROOT / "EndovascularSurgery", "weiss", "opt_label_weiss"
    ),
    "jhu": DatasetConfig("jhu", ROOT / "Orthopedic", "jhu", "opt_label_jhu"),
    "cardio": DatasetConfig(
        "cardio", ROOT / "InterventionalCardiology", "cardiac", "opt_label_cardiac"
    ),
}


def _apply_weiss_resolution_rule(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    if frame.shape == (256, 256):
        return frame
    return cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)


def _apply_cardiac_resolution_rule(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    if frame.shape == (512, 512):
        return cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
    return frame


def _collect_jhu_sequences(
    repo: Path, data_mod: ModuleType
) -> List[Tuple[str, List[Path]]]:
    base = repo
    if (repo / "jhu").exists():
        base = repo / "jhu"
    elif (repo / "JHU").exists():
        base = repo / "JHU"

    test_patients = set(getattr(data_mod, "JHU_TEST_PATIENTS", []))
    patients = sorted([p for p in base.glob("cadaver_*") if p.is_dir()])
    trainval = [p for p in patients if p.name not in test_patients]

    def _frame_key(p: Path):
        stem = p.stem.split("_")[0]
        try:
            return int(stem), str(p)
        except ValueError:
            return 10**9, str(p)

    out: List[Tuple[str, List[Path]]] = []
    for patient_dir in trainval:
        fluoro = patient_dir / "fluoro"
        if not fluoro.exists():
            continue
        preview_dirs = sorted([p for p in fluoro.rglob("preview") if p.is_dir()])
        for preview_dir in preview_dirs:
            files = [
                p
                for p in preview_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
            files = sorted(files, key=_frame_key)
            if not files:
                continue
            rel = preview_dir.relative_to(patient_dir / "fluoro")
            seq_key = f"{patient_dir.name}__{str(rel).replace('/', '_')}"
            out.append((seq_key, files))
    return sorted(out, key=lambda x: x[0])


def _collect_cardiac_sequences(
    repo: Path, data_mod: ModuleType, split: str
) -> List[Tuple[str, Path]]:
    CARDIAC_LEFT = "Left_Dominance"
    CARDIAC_RIGHT = "Right_Dominance"
    seed = 1337
    ratios = (0.7, 0.15, 0.15)

    data_root = getattr(data_mod, "CARDIAC_DATA_ROOT", "cardio_data")
    anon_root = getattr(data_mod, "CARDIAC_ANON_ROOT", "anonymous_syntax_holdout")

    if (repo / data_root / anon_root).exists():
        base = repo / data_root / anon_root
    elif (repo / anon_root).exists():
        base = repo / anon_root
    else:
        base = repo / data_root / anon_root

    studies: List[Path] = []
    for dom in (CARDIAC_LEFT, CARDIAC_RIGHT):
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
    if not studies:
        return []

    rng = random.Random(seed)
    ordered = list(studies)
    rng.shuffle(ordered)
    n = len(ordered)

    train_ratio, val_ratio, _test_ratio = ratios
    train_n = max(1, int(n * train_ratio))
    val_n = max(1, int(n * val_ratio))
    test_n = max(1, n - train_n - val_n)
    if train_n + val_n + test_n != n:
        test_n = n - train_n - val_n
        if test_n < 1 and n > 2:
            test_n = 1
            val_n = max(0, n - train_n - test_n)

    train_end = min(n, train_n)
    val_end = min(n, train_end + val_n)
    if split == "training":
        picked = ordered[:train_end]
    elif split == "validation":
        picked = ordered[train_end:val_end]
    else:
        picked = ordered[val_end:]

    seqs: List[Tuple[str, Path]] = []
    for study in picked:
        for view in ("LCA", "RCA"):
            view_dir = study / view
            if not view_dir.exists():
                continue
            for path in sorted(view_dir.glob("*.npz")):
                seq_key = f"{path.relative_to(base)}".replace("/", "__").replace(
                    ".npz", ""
                )
                seqs.append((seq_key, path))
    return sorted(seqs, key=lambda x: x[0])


def _load_cardiac_sequence(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as arr_npz:
        arr = arr_npz["pixel_array"]
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr.astype(np.float32, copy=False)


def _run_weiss(cfg: DatasetConfig, data_mod: ModuleType, args) -> None:
    out_root = cfg.repo / cfg.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    weiss_splits = getattr(data_mod, "WEISS_SPLITS")
    combined_ranges = list(weiss_splits["training"]) + list(weiss_splits["validation"])

    def _collect_indices(dataset_name: str) -> List[int]:
        indices: List[int] = []
        for name, a, b in combined_ranges:
            if name != dataset_name:
                continue
            indices.extend(list(range(int(a), int(b) + 1)))
        return sorted(indices)

    info_map = {
        "T1T2": {"path": "weiss/T1T2.hdf5", "dataset": "train_img"},
        "T3T6": {"path": "weiss/T3-T6.hdf5", "dataset": "test"},
    }

    index_map: Dict[str, Dict[int, int]] = {}
    file_map: Dict[str, str] = {}
    dataset_map: Dict[str, str] = {}

    for dataset_name, info in info_map.items():
        indices = _collect_indices(dataset_name)
        if not indices:
            continue

        h5_path = cfg.repo / info["path"]
        with h5py.File(h5_path, "r") as h5:
            ds = h5[info["dataset"]]

            global_by_local = list(indices)

            def get_clean_local(local_idx: int) -> np.ndarray:
                arr = _select_middle_channel(
                    ds[int(global_by_local[int(local_idx)])]
                ).astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
                return _apply_weiss_resolution_rule(arr)

            out_name = f"{dataset_name}_opt_label.hdf5"
            out_path = out_root / out_name

            local_count = len(global_by_local)
            local_indices, _ = run_sequence(
                seq_key=dataset_name,
                num_frames=local_count,
                get_clean_frame=get_clean_local,
                data_mod=data_mod,
                out_path=out_path,
                params=args.fusion_params,
                max_frames=args.max_frames,
                no_black_border_crop=bool(args.no_black_border_crop),
                seed=int(args.seed),
            )
            index_map[dataset_name] = {
                int(global_by_local[i]): int(i) for i in local_indices
            }
            file_map[dataset_name] = out_name
            dataset_map[dataset_name] = "opt_label"

    index_path = out_root / "opt_label_index.json"
    index_path.write_text(
        json.dumps(
            {"map": index_map, "files": file_map, "datasets": dataset_map}, indent=2
        )
    )
    print(f"[done] weiss index: {index_path}")


def _run_jhu(cfg: DatasetConfig, data_mod: ModuleType, args) -> None:
    out_root = cfg.repo / cfg.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    seq_items = _collect_jhu_sequences(cfg.repo, data_mod)
    if not seq_items:
        raise RuntimeError(f"No JHU preview sequences found under {cfg.repo}")

    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)

    index_map: Dict[str, Dict[int, int]] = {}
    file_map: Dict[str, str] = {}
    dataset_map: Dict[str, str] = {}

    for seq_idx, (seq_key, files) in enumerate(seq_items):
        if seq_idx % num_shards != shard_index:
            continue
        print(f"[jhu] [{seq_idx + 1}/{len(seq_items)}] {seq_key}")

        def get_clean(i: int) -> np.ndarray:
            img = Image.open(files[int(i)]).convert("L")
            arr = np.array(img, dtype=np.float32)
            return arr

        out_name = f"{seq_key}_opt_label.hdf5"
        out_path = out_root / out_name
        local_indices, _ = run_sequence(
            seq_key=seq_key,
            num_frames=len(files),
            get_clean_frame=get_clean,
            data_mod=data_mod,
            out_path=out_path,
            params=args.fusion_params,
            max_frames=args.max_frames,
            no_black_border_crop=bool(args.no_black_border_crop),
            seed=int(args.seed),
        )
        index_map[seq_key] = {int(i): int(i) for i in local_indices}
        file_map[seq_key] = out_name
        dataset_map[seq_key] = "opt_label"

    if num_shards == 1:
        index_name = "opt_label_index.json"
    else:
        index_name = f"opt_label_index.shard{shard_index}of{num_shards}.json"
    index_path = out_root / index_name
    index_path.write_text(
        json.dumps(
            {"map": index_map, "files": file_map, "datasets": dataset_map}, indent=2
        )
    )
    print(f"[done] jhu index: {index_path}")


def _run_cardiac(cfg: DatasetConfig, data_mod: ModuleType, args) -> None:
    out_root = cfg.repo / cfg.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    seq_items: List[Tuple[str, Path]] = []
    for split in ("training", "validation"):
        seq_items.extend(_collect_cardiac_sequences(cfg.repo, data_mod, split))
    seq_items = sorted(seq_items, key=lambda x: x[0])
    if not seq_items:
        raise RuntimeError(f"No cardiac sequences found under {cfg.repo}")

    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)

    index_map: Dict[str, Dict[int, int]] = {}
    file_map: Dict[str, str] = {}
    dataset_map: Dict[str, str] = {}

    for seq_idx, (seq_key, npz_path) in enumerate(seq_items):
        if seq_idx % num_shards != shard_index:
            continue
        print(f"[cardio] [{seq_idx + 1}/{len(seq_items)}] {seq_key}")
        frames = _load_cardiac_sequence(npz_path)

        def get_clean(i: int) -> np.ndarray:
            fr = _select_middle_channel(frames[int(i)])
            fr = _apply_cardiac_resolution_rule(fr)
            if fr.max() > 1.5:
                fr = fr / 255.0
            return fr.astype(np.float32)

        out_name = f"{seq_key}_opt_label.hdf5"
        out_path = out_root / out_name
        local_indices, _ = run_sequence(
            seq_key=seq_key,
            num_frames=int(frames.shape[0]),
            get_clean_frame=get_clean,
            data_mod=data_mod,
            out_path=out_path,
            params=args.fusion_params,
            max_frames=args.max_frames,
            no_black_border_crop=bool(args.no_black_border_crop),
            seed=int(args.seed),
        )
        index_map[seq_key] = {int(i): int(i) for i in local_indices}
        file_map[seq_key] = out_name
        dataset_map[seq_key] = "opt_label"

    if num_shards == 1:
        index_name = "opt_label_index.json"
    else:
        index_name = f"opt_label_index.shard{shard_index}of{num_shards}.json"
    index_path = out_root / index_name
    index_path.write_text(
        json.dumps(
            {"map": index_map, "files": file_map, "datasets": dataset_map}, indent=2
        )
    )
    print(f"[done] cardiac index: {index_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate final motion-adaptive opt-flow pseudo labels."
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASET_CONFIGS),
        default=["weiss", "jhu", "cardio"],
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--max_frames", type=int, default=None, help="Optional debug cap per sequence."
    )
    p.add_argument("--no_black_border_crop", action="store_true")

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)

    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--search_radius", type=int, default=10)
    p.add_argument("--max_neighbors", type=int, default=8)
    p.add_argument("--sim_threshold", type=float, default=0.10)

    p.add_argument("--flow", choices=["dis", "tvl1"], default="dis")
    p.add_argument("--no_vst", action="store_true")
    p.add_argument("--no_ecc", action="store_true")
    p.add_argument("--stages", type=int, choices=[1, 2], default=2)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    _set_seed(int(args.seed))

    params = FusionParams(
        radius=int(args.radius),
        search_radius=int(args.search_radius),
        max_neighbors=int(args.max_neighbors),
        sim_threshold=float(args.sim_threshold),
        flow=str(args.flow),
        use_vst=not bool(args.no_vst),
        use_ecc=not bool(args.no_ecc),
        stages=int(args.stages),
    )
    args.fusion_params = params

    for name in args.datasets:
        cfg = DATASET_CONFIGS[str(name)]
        print(f"\n[info] dataset={cfg.name} repo={cfg.repo}")

        data_py = cfg.repo / "data.py"
        if not data_py.exists():
            raise FileNotFoundError(
                f"Missing {data_py}. Run this script from the project root."
            )

        data_mod = _load_module(data_py, f"final_new_opt_label_data_{cfg.name}")

        if cfg.name == "weiss":
            _run_weiss(cfg, data_mod, args)
        elif cfg.name == "jhu":
            _run_jhu(cfg, data_mod, args)
        elif cfg.name == "cardio":
            _run_cardiac(cfg, data_mod, args)
        else:
            raise ValueError(f"Unsupported dataset: {cfg.name}")


if __name__ == "__main__":
    main()
