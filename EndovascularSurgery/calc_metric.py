import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DEFAULT_LOW_DOSE_CFG, DenoiseDataset, get_noise_level
from model import BM3D, FFDNet, FastDVDnet, NLM
from train import build_model


def _fmt_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


DEFAULT_NOISE_CFG = dict(DEFAULT_LOW_DOSE_CFG)
DEFAULT_MODELS = ["unet", "fdk"]
DEFAULT_MODES = ["n2c", "n2v", "r2r", "opt+r"]
NON_TRAINABLE_MODELS = {"bm3d", "nlm"}
SUPERVISED_MODELS = {
    "unet",
    "unet_tmp",
    "redcnn",
    "red-cnn",
    "wganvgg",
    "wgan-vgg",
    "dncnn",
    "ffdnet",
    "nafnet",
    "edvr",
    "fastdvdnet",
}


def _infer_dataset_tag(_root: Path, _dataset_kind: str = "auto") -> str:
    return "weiss"


def _normalize_model_name(name: str) -> str:
    name = name.lower()
    if name == "red-cnn":
        return "redcnn"
    if name == "wgan-vgg":
        return "wganvgg"
    if name in {"unet-tmp", "unettemp", "unettmp"}:
        return "unet_tmp"
    return name


def _normalize_mode_alias(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"opt+r"}:
        return "opt+r"
    return mode


def _normalize_entry_mode(name: str) -> str:
    parts = str(name).split("-")
    if len(parts) > 1:
        parts[1] = _normalize_mode_alias(parts[1])
    return "-".join(parts)


def _with_dose_peak(name: str, dose_tag: str, peak_tag: str) -> str:
    name = _normalize_entry_mode(name)
    parts = name.split("-")
    if len(parts) >= 4:
        return name
    return f"{name}-{dose_tag}-{peak_tag}"


def _resolve_dose_peak(
    name: str, dose_fraction: float, peak: float
) -> tuple[str, float, float, str, str]:
    name = (
        _normalize_entry_mode(name)
        .lower()
        .replace("wgan-vgg", "wganvgg")
        .replace("red-cnn", "redcnn")
        .replace("unet-tmp", "unet_tmp")
        .replace("unettemp", "unet_tmp")
        .replace("unettmp", "unet_tmp")
    )
    parts = name.split("-")
    base_model = _normalize_model_name(parts[0])
    dose_value = float(dose_fraction)
    peak_value = float(peak)
    if len(parts) >= 4:
        try:
            dose_value = float(parts[-2])
            peak_value = float(parts[-1])
        except ValueError:
            pass
    dose_tag = _fmt_float(dose_value)
    peak_tag = _fmt_float(peak_value)
    return base_model, dose_value, peak_value, dose_tag, peak_tag


def _parse_csv(arg: str | None) -> list[str]:
    if not arg:
        return []
    return [item.strip() for item in arg.split(",") if item.strip()]


def _lock_noise_args(args: argparse.Namespace) -> None:
    for key, default in DEFAULT_NOISE_CFG.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        changed = (
            abs(float(current) - float(default)) > 1e-12
            if isinstance(default, (int, float))
            else current != default
        )
        if changed:
            print(
                f"[info] Ignoring --{key}={current}; using data.DEFAULT_LOW_DOSE_CFG[{key!r}]={default}."
            )
        setattr(args, key, default)


SUMMARY_FIELDS = [
    "entry",
    "model",
    "mode",
    "psnr",
    "ssim",
    "latency_ms_per_frame",
    "dose_fraction",
    "peak",
]


def build_model_entries(
    models_arg: str | None,
    modes_arg: str | None,
    dose_fraction: float,
    peak: float,
) -> list[str]:
    dose_tag = _fmt_float(dose_fraction)
    peak_tag = _fmt_float(peak)
    models_list = _parse_csv(models_arg) or list(DEFAULT_MODELS)
    modes_list = [
        _normalize_mode_alias(m) for m in (_parse_csv(modes_arg) or list(DEFAULT_MODES))
    ]

    entries: list[str] = []
    for raw_model in models_list:
        raw_model = (
            _normalize_entry_mode(raw_model)
            .lower()
            .replace("wgan-vgg", "wganvgg")
            .replace("red-cnn", "redcnn")
            .replace("unet-tmp", "unet_tmp")
            .replace("unettemp", "unet_tmp")
            .replace("unettmp", "unet_tmp")
        )
        parts = raw_model.split("-")
        base_model = _normalize_model_name(parts[0])

        if len(parts) >= 4:
            entries.append(raw_model)
            continue
        if len(parts) == 3:
            entries.append(f"{raw_model}-{peak_tag}")
            continue
        if len(parts) == 2:
            entries.append(f"{raw_model}-{dose_tag}-{peak_tag}")
            continue

        if base_model in NON_TRAINABLE_MODELS:
            entries.append(f"{base_model}-n2c-{dose_tag}-{peak_tag}")
            continue

        if base_model in SUPERVISED_MODELS and base_model not in {
            "unet",
            "unet_tmp",
            "fdk",
        }:
            entries.append(f"{base_model}-n2c-{dose_tag}-{peak_tag}")
            continue

        for mode in modes_list:
            entries.append(f"{base_model}-{mode}-{dose_tag}-{peak_tag}")

    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return deduped


def _ckpt_candidates(
    dataset_tag: str, name: str, dose_tag: str, peak_tag: str
) -> list[Path]:
    dataset_dir = Path("saved_model") / dataset_tag
    parts = name.split("-")
    if len(parts) == 4 and parts[1] == "pretrain":
        dataset_dir = Path("saved_model") / f"pre_{dataset_tag}"
    lookup_name = _with_dose_peak(name, dose_tag, peak_tag)
    return [
        dataset_dir / f"{lookup_name}.pt",
        dataset_dir / f"{name}.pt",
        Path("saved_model") / f"{lookup_name}.pt",
        Path("saved_model") / f"{name}.pt",
    ]


def _gaussian_window(window_size, sigma, channel, device, dtype):
    coords = (
        torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    )
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    window = window.repeat(channel, 1, 1, 1)
    return window


def ssim_torch(x, y, window_size=11, sigma=1.5, data_range=1.0):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    _b, c, _h, _w = x.shape
    window = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    mu1 = torch.nn.functional.conv2d(x, window, padding=window_size // 2, groups=c)
    mu2 = torch.nn.functional.conv2d(y, window, padding=window_size // 2, groups=c)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2
    sigma1_sq = (
        torch.nn.functional.conv2d(x * x, window, padding=window_size // 2, groups=c)
        - mu1_sq
    )
    sigma2_sq = (
        torch.nn.functional.conv2d(y * y, window, padding=window_size // 2, groups=c)
        - mu2_sq
    )
    sigma12 = (
        torch.nn.functional.conv2d(x * y, window, padding=window_size // 2, groups=c)
        - mu12
    )
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def psnr_torch(x, y, data_range=1.0):
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3))
    eps = 1e-12
    return 10.0 * torch.log10((data_range**2) / (mse + eps))


def _extract_model_kwargs(model_name: str, ckpt_args: dict, burst_size: int) -> dict:
    model_name = _normalize_model_name(model_name)
    if model_name == "fdk":
        return {
            "width": int(ckpt_args.get("fdk_width", 32)),
            "blocks": int(ckpt_args.get("fdk_blocks", 4)),
            "radius": int(ckpt_args.get("fdk_radius", 1)),
            "down": 1,
            "eps": float(ckpt_args.get("fdk_eps", 1e-4)),
            "temporal_input_fusion": True,
            "temporal_center_only": True,
            "temporal_attn_temp": 0.1,
            "use_var": bool(ckpt_args.get("fdk_use_var", False)),
            "disable_kalman": bool(ckpt_args.get("fdk_disable_kalman", False)),
        }
    if model_name == "unet":
        return {
            "base_channels": int(ckpt_args.get("unet_base_channels", 16)),
            "depth": int(ckpt_args.get("unet_depth", 3)),
        }
    if model_name == "unet_tmp":
        return {
            "base_channels": int(ckpt_args.get("unet_base_channels", 16)),
            "depth": int(ckpt_args.get("unet_depth", 3)),
            "temporal_attn_temp": float(
                ckpt_args.get("unet_tmp_temporal_attn_temp", 0.1)
            ),
            "temporal_input_fusion": bool(
                ckpt_args.get("unet_tmp_temporal_input_fusion", True)
            ),
            "temporal_center_only": bool(
                ckpt_args.get("unet_tmp_temporal_center_only", True)
            ),
        }
    if model_name == "dncnn":
        return {
            "depth": int(ckpt_args.get("dncnn_depth", 17)),
            "features": int(ckpt_args.get("dncnn_features", 64)),
        }
    if model_name == "fastdvdnet":
        return {
            "num_frames": int(max(1, burst_size)),
            "default_noise_level": float(ckpt_args.get("noise_level", 0.03)),
            "base_channels": int(ckpt_args.get("fastdvdnet_base_channels", 32)),
        }
    if model_name == "redcnn":
        return {
            "channels": int(ckpt_args.get("redcnn_channels", 96)),
            "num_layers": int(ckpt_args.get("redcnn_layers", 5)),
        }
    if model_name == "wganvgg":
        return {
            "base_channels": int(ckpt_args.get("wgan_base_channels", 32)),
            "num_blocks": int(ckpt_args.get("wgan_blocks", 8)),
        }
    if model_name == "ffdnet":
        return {
            "features": int(ckpt_args.get("ffdnet_features", 64)),
            "depth": int(ckpt_args.get("ffdnet_depth", 15)),
            "default_noise_level": float(ckpt_args.get("noise_level", 0.03)),
        }
    if model_name == "nafnet":
        return {
            "width": int(ckpt_args.get("nafnet_width", 32)),
            "blocks": int(ckpt_args.get("nafnet_blocks", 6)),
        }
    if model_name == "edvr":
        return {
            "num_frames": int(max(1, burst_size)),
            "channels": int(ckpt_args.get("edvr_channels", 32)),
            "num_blocks": int(ckpt_args.get("edvr_blocks", 10)),
        }
    return {}


def load_model_entry(
    name: str,
    device: torch.device,
    dataset_tag: str,
    dose_fraction: float,
    peak: float,
):
    base_model, dose_value, peak_value, dose_tag, peak_tag = _resolve_dose_peak(
        name, dose_fraction, peak
    )
    if base_model in NON_TRAINABLE_MODELS:
        lookup_name = f"{base_model}-n2c-{dose_tag}-{peak_tag}"
        noise_cfg = dict(DEFAULT_NOISE_CFG)
        if base_model == "bm3d":
            model = BM3D(sigma_psd=get_noise_level(noise_cfg))
        else:
            model = NLM()
        eval_opts = {
            "burst_align": False,
            "burst_align_max_shift": 10.0,
            "burst_causal": False,
            "burst_target": "center",
            "target_pos": None,
        }
        return model.to(device), "n2c", noise_cfg, 1, 1, lookup_name, eval_opts

    lookup_name = _with_dose_peak(name, dose_tag, peak_tag)
    ckpt_path = next(
        (
            p
            for p in _ckpt_candidates(dataset_tag, name, dose_tag, peak_tag)
            if p.exists()
        ),
        None,
    )
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found for {name} under saved_model/{dataset_tag}"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    noise_cfg = dict(DEFAULT_NOISE_CFG)

    parts = lookup_name.split("-")
    model_name = _normalize_model_name(parts[0])
    mode = _normalize_mode_alias(
        parts[1] if len(parts) > 1 else ckpt_args.get("mode", "n2c")
    )
    in_channels = int(ckpt_args.get("in_channels", 1))
    burst_size = int(ckpt.get("burst_size", ckpt_args.get("video_burst", 1)))

    model_kwargs = _extract_model_kwargs(model_name, ckpt_args, burst_size)
    model = build_model(model_name, in_channels=in_channels, **model_kwargs)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[warn] Missing keys while loading {name}: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading {name}: {len(unexpected)}")

    if model_name in {"fdk", "unet_tmp"} and burst_size > 1:
        eval_opts = {
            "burst_align": False,
            "burst_align_max_shift": 10.0,
            "burst_causal": True,
            "burst_target": "last",
            "target_pos": burst_size - 1,
        }
    else:
        eval_opts = {
            "burst_align": False,
            "burst_align_max_shift": 10.0,
            "burst_causal": False,
            "burst_target": "center",
            "target_pos": None,
        }

    model = model.to(device)
    return model, mode, noise_cfg, in_channels, burst_size, lookup_name, eval_opts


def _select_target_if_burst(
    x: torch.Tensor, burst_size: int, target_pos: int | None = None
) -> torch.Tensor:
    if x.dim() == 5:
        pos = burst_size // 2 if target_pos is None else int(target_pos)
        pos = max(0, min(pos, x.size(1) - 1))
        return x[:, pos]
    return x


def _run_model(
    model: torch.nn.Module, inp: torch.Tensor, noise_cfg: dict
) -> torch.Tensor:
    if isinstance(model, (FastDVDnet, FFDNet)):
        noise_level = get_noise_level(noise_cfg)
        h, w = inp.size(-2), inp.size(-1)
        noise_map = torch.full(
            (inp.size(0), 1, h, w), noise_level, device=inp.device, dtype=inp.dtype
        )
        return model(inp, noise_map)
    return model(inp)


def eval_split(
    model: torch.nn.Module,
    split: str,
    noise_cfg: dict,
    in_channels: int,
    device: torch.device,
    root: Path,
    burst_size: int,
    *,
    dataset_kind: str,
    burst_align: bool = False,
    burst_align_max_shift: float = 10.0,
    burst_causal: bool = False,
    burst_target: str = "center",
    target_pos: int | None = None,
):
    patch_size = None if split == "test" else 256
    ds = DenoiseDataset(
        root,
        split,
        patch_size,
        "n2c",
        in_channels,
        noise_cfg,
        dataset_kind=dataset_kind,
        burst_size=burst_size,
        burst_align=burst_align,
        burst_align_max_shift=burst_align_max_shift,
        burst_causal=burst_causal,
        burst_target=burst_target,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    psnr_vals = []
    ssim_vals = []
    model.eval()
    with torch.no_grad():
        for inp, target in loader:
            inp = inp.to(device)
            target = target.to(device)
            target_frame = _select_target_if_burst(target, burst_size, target_pos)
            output = _run_model(model, inp, noise_cfg)
            output = _select_target_if_burst(output, burst_size, target_pos)
            output = torch.clamp(output, 0.0, 1.0)

            psnr_vals.append(psnr_torch(output, target_frame).item())
            ssim_vals.append(ssim_torch(output, target_frame).item())

    psnr_mean = sum(psnr_vals) / max(1, len(psnr_vals))
    ssim_mean = sum(ssim_vals) / max(1, len(ssim_vals))
    return psnr_mean, ssim_mean


def _bench_dtype(model: torch.nn.Module) -> torch.dtype:
    for p in model.parameters():
        return p.dtype
    return torch.float32


def _latency_signature(
    model: torch.nn.Module,
    *,
    in_channels: int,
    burst_size: int,
    h: int,
    w: int,
    batch_size: int,
) -> tuple:
    param_shapes = tuple((name, tuple(p.shape)) for name, p in model.named_parameters())
    buffer_shapes = tuple((name, tuple(b.shape)) for name, b in model.named_buffers())
    return (
        model.__class__.__name__,
        int(in_channels),
        int(burst_size),
        int(h),
        int(w),
        int(batch_size),
        param_shapes,
        buffer_shapes,
    )


def _merge_summary_rows(summary_path: Path, fresh_rows: list[dict]) -> list[dict]:
    order: list[str] = []
    merged: dict[str, dict] = {}

    if summary_path.exists():
        with summary_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = row.get("entry", "")
                if not entry:
                    continue
                if (
                    entry.startswith("sdd2022-")
                    or str(row.get("model", "")).lower() == "sdd2022"
                ):
                    continue
                order.append(entry)
                merged[entry] = {k: row.get(k, "") for k in SUMMARY_FIELDS}

    for row in fresh_rows:
        entry = str(row.get("entry", ""))
        if not entry:
            continue
        if entry not in merged:
            order.append(entry)
        merged[entry] = row

    return [merged[e] for e in order if e in merged]


def benchmark_latency(
    model: torch.nn.Module,
    noise_cfg: dict,
    in_channels: int,
    burst_size: int,
    device: torch.device,
    *,
    h: int,
    w: int,
    batch_size: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> float:
    model.eval()
    dtype = _bench_dtype(model)
    is_classical = isinstance(model, (BM3D, NLM))
    bench_device = torch.device("cpu") if is_classical else device
    model = model.to(bench_device)
    prev_bench = torch.backends.cudnn.benchmark
    if bench_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        if burst_size > 1:
            inp = torch.rand(
                (batch_size, burst_size, in_channels, h, w),
                device=bench_device,
                dtype=dtype,
            )
        else:
            inp = torch.rand(
                (batch_size, in_channels, h, w), device=bench_device, dtype=dtype
            )

        with torch.inference_mode():
            for _ in range(max(0, warmup)):
                _ = _run_model(model, inp, noise_cfg)
            if bench_device.type == "cuda":
                torch.cuda.synchronize(bench_device)

            per_frame_ms: list[float] = []
            for _ in range(max(1, repeats)):
                if bench_device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(bench_device)
                    start.record()
                    out = None
                    for _ in range(max(1, iters)):
                        out = _run_model(model, inp, noise_cfg)
                    end.record()
                    torch.cuda.synchronize(bench_device)
                    elapsed_ms = float(start.elapsed_time(end))
                else:
                    t0 = time.perf_counter()
                    out = None
                    for _ in range(max(1, iters)):
                        out = _run_model(model, inp, noise_cfg)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                out_frames = 1 if out is None or out.dim() != 5 else int(out.size(1))
                per_frame_ms.append(elapsed_ms / (max(1, iters) * max(1, out_frames)))

        return float(np.median(per_frame_ms))
    finally:
        if bench_device.type == "cuda":
            torch.backends.cudnn.benchmark = prev_bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset_kind", default="weiss", choices=["weiss"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Comma-separated string or space-separated list of model entries",
    )
    parser.add_argument("--modes", default=None)
    parser.add_argument(
        "--dose_fraction", type=float, default=DEFAULT_NOISE_CFG["dose_fraction"]
    )
    parser.add_argument("--peak", type=float, default=DEFAULT_NOISE_CFG["peak"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--latency_warmup", type=int, default=30)
    parser.add_argument(
        "--latency_samples",
        type=int,
        default=200,
        help="Benchmark iterations per repeat",
    )
    parser.add_argument("--latency_repeats", type=int, default=5)
    parser.add_argument("--latency_height", type=int, default=256)
    parser.add_argument("--latency_width", type=int, default=256)
    parser.add_argument("--latency_batch_size", type=int, default=1)
    args = parser.parse_args()
    _lock_noise_args(args)

    if args.seed is not None:
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    root_path = Path(args.root)
    dataset_tag = _infer_dataset_tag(root_path, args.dataset_kind)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir_default = Path("saved_metric") / dataset_tag
    out_dir_default.mkdir(parents=True, exist_ok=True)

    models_arg = None
    if args.models:
        if isinstance(args.models, list):
            models_arg = ",".join(args.models)
        else:
            models_arg = args.models

    summary_rows = []
    latency_cache: dict[tuple, float] = {}

    for name in build_model_entries(
        models_arg, args.modes, args.dose_fraction, args.peak
    ):
        try:
            model, mode, noise_cfg, in_channels, burst_size, lookup_name, eval_opts = (
                load_model_entry(
                    name,
                    device,
                    dataset_tag,
                    args.dose_fraction,
                    args.peak,
                )
            )
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        if "pretrain" in name:
            out_dir = Path("saved_metric") / f"pre_{dataset_tag}"
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = out_dir_default

        psnr, ssim = eval_split(
            model,
            "test",
            noise_cfg,
            in_channels,
            device,
            root_path,
            burst_size,
            dataset_kind=dataset_tag,
            burst_align=bool(eval_opts.get("burst_align", False)),
            burst_align_max_shift=float(eval_opts.get("burst_align_max_shift", 10.0)),
            burst_causal=bool(eval_opts.get("burst_causal", False)),
            burst_target=str(eval_opts.get("burst_target", "center")),
            target_pos=eval_opts.get("target_pos"),
        )
        sig = _latency_signature(
            model,
            in_channels=in_channels,
            burst_size=burst_size,
            h=int(args.latency_height),
            w=int(args.latency_width),
            batch_size=int(args.latency_batch_size),
        )
        if sig in latency_cache:
            latency_ms = latency_cache[sig]
        else:
            latency_ms = benchmark_latency(
                model,
                noise_cfg,
                in_channels,
                burst_size,
                device,
                h=int(args.latency_height),
                w=int(args.latency_width),
                batch_size=int(args.latency_batch_size),
                warmup=int(args.latency_warmup),
                iters=int(args.latency_samples),
                repeats=int(args.latency_repeats),
            )
            latency_cache[sig] = latency_ms

        out_path = out_dir / f"{lookup_name}.csv"
        rows = [
            {
                "split": "test",
                "psnr": psnr,
                "ssim": ssim,
                "latency_ms_per_frame": latency_ms,
            }
        ]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["split", "psnr", "ssim", "latency_ms_per_frame"]
            )
            writer.writeheader()
            writer.writerows(rows)

        summary_rows.append(
            {
                "entry": lookup_name,
                "model": _normalize_model_name(lookup_name.split("-")[0]),
                "mode": mode,
                "psnr": psnr,
                "ssim": ssim,
                "latency_ms_per_frame": latency_ms,
                "dose_fraction": noise_cfg.get("dose_fraction", args.dose_fraction),
                "peak": noise_cfg.get("peak", args.peak),
            }
        )
        print(f"Saved metrics to {out_path}")

    if summary_rows:
        summary_path = out_dir_default / f"summary_{dataset_tag}.csv"
        merged_rows = _merge_summary_rows(summary_path, summary_rows)
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=SUMMARY_FIELDS,
            )
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
