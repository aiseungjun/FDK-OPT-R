from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path


def _fmt_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ckpt_entry(model: str, mode: str, dose_fraction: float, peak: float) -> str:
    return f"{model}-{mode}-{_fmt_float(dose_fraction)}-{_fmt_float(peak)}"


def _normalize_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"opt+r"}:
        return "opt+r"
    return mode


def _read_metric(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty metric csv: {path}")
    return rows[0]


ROOT = Path(__file__).resolve().parent
DATASET_CONFIGS = {
    "weiss": {
        "repo": ROOT / "EndovascularSurgery",
        "kind": "weiss",
        "opt_label_root": "opt_label_weiss",
    },
    "jhu": {
        "repo": ROOT / "Orthopedic",
        "kind": "jhu",
        "opt_label_root": "opt_label_jhu",
    },
    "cardio": {
        "repo": ROOT / "InterventionalCardiology",
        "kind": "cardiac",
        "opt_label_root": "opt_label_cardiac",
    },
}

MODEL_SPECS = {
    "unet-small": {"train_model": "unet", "unet_depth": 2, "unet_base_channels": 8},
    "unet-base": {"train_model": "unet", "unet_depth": 3, "unet_base_channels": 8},
    "unet-big": {"train_model": "unet", "unet_depth": 3, "unet_base_channels": 16},
    "unet-tmp-small": {
        "train_model": "unet_tmp",
        "unet_depth": 2,
        "unet_base_channels": 8,
    },
    "unet-tmp-base": {
        "train_model": "unet_tmp",
        "unet_depth": 3,
        "unet_base_channels": 8,
    },
    "unet-tmp-big": {
        "train_model": "unet_tmp",
        "unet_depth": 3,
        "unet_base_channels": 16,
    },
    "fdk-small": {
        "train_model": "fdk",
        "fdk_width": 8,
        "fdk_blocks": 3,
        "fdk_radius": 1,
        "fdk_down": 1,
    },
    "fdk-base": {
        "train_model": "fdk",
        "fdk_width": 16,
        "fdk_blocks": 4,
        "fdk_radius": 1,
        "fdk_down": 1,
    },
    "fdk-big": {
        "train_model": "fdk",
        "fdk_width": 32,
        "fdk_blocks": 4,
        "fdk_radius": 1,
        "fdk_down": 1,
    },
    "fdk": {
        "train_model": "fdk",
        "fdk_width": 32,
        "fdk_blocks": 4,
        "fdk_radius": 1,
        "fdk_down": 1,
    },
}

DEFAULT_MODELS = [
    "unet-small",
    "unet-base",
    "unet-big",
    "unet-tmp-small",
    "unet-tmp-base",
    "unet-tmp-big",
    "fdk-small",
    "fdk-base",
    "fdk-big",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate temporal-power ablations and export CSV summaries."
    )
    parser.add_argument("--workspace", default="./ablation_temp_power_runs")
    parser.add_argument(
        "--datasets", nargs="*", choices=list(DATASET_CONFIGS), default=["weiss", "jhu"]
    )
    parser.add_argument(
        "--models", nargs="*", choices=list(MODEL_SPECS), default=DEFAULT_MODELS
    )
    parser.add_argument(
        "--mode", choices=["n2c", "opt+r"], default="opt+r"
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--opt_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--dose_fraction",
        type=float,
        default=None,
        help="Deprecated override. Corrupt settings are always read from each dataset's data.DEFAULT_LOW_DOSE_CFG.",
    )
    parser.add_argument("--opt_burst", type=int, default=5)

    parser.add_argument("--latency_warmup", type=int, default=30)
    parser.add_argument("--latency_samples", type=int, default=200)
    parser.add_argument("--latency_repeats", type=int, default=5)
    parser.add_argument("--latency_height", type=int, default=256)
    parser.add_argument("--latency_width", type=int, default=256)
    return parser


def _load_default_noise_cfg(repo: Path) -> dict:
    data_py = repo / "data.py"
    if not data_py.exists():
        raise FileNotFoundError(f"Missing data.py under dataset repo: {repo}")
    spec = importlib.util.spec_from_file_location(
        f"ablation_data_{repo.name}", str(data_py)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {data_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "DEFAULT_LOW_DOSE_CFG", None)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"DEFAULT_LOW_DOSE_CFG is missing or invalid in {data_py}")
    if "dose_fraction" not in cfg or "peak" not in cfg or "sigma_read" not in cfg:
        raise RuntimeError(
            f"DEFAULT_LOW_DOSE_CFG must include dose_fraction/peak/sigma_read in {data_py}"
        )
    if cfg.get("peak") is None or cfg.get("sigma_read") is None:
        raise RuntimeError(
            f"Unset peak/sigma_read in {data_py}. Run `python peak_sigma_estimate.py` under paper_code, "
            "then copy estimate_peak and estimate_sigma_read into each dataset data.py."
        )
    return dict(cfg)


def _ensure_opt_label_ready(
    dataset_name: str, cfg: dict, args: argparse.Namespace
) -> None:
    if _normalize_mode(args.mode) != "opt+r":
        return

    repo = Path(cfg["repo"])
    label_root = repo / str(cfg["opt_label_root"])
    index_path = label_root / "opt_label_index.json"
    has_h5 = any(label_root.glob("*_opt_label.hdf5"))
    if not index_path.exists() or not has_h5:
        raise FileNotFoundError(
            f"Missing precomputed opt labels for {dataset_name}: expected index/hdf5 under {label_root}"
        )


def _run_one(
    dataset_name: str,
    model_key: str,
    cfg: dict,
    args: argparse.Namespace,
    workspace: Path,
) -> dict[str, str]:
    spec = MODEL_SPECS[model_key]
    repo = Path(cfg["repo"])
    dataset_kind = str(cfg["kind"])
    noise_cfg = dict(cfg["noise_cfg"])
    dose_fraction = float(noise_cfg["dose_fraction"])
    peak = float(noise_cfg["peak"])
    run_dir = workspace / dataset_name / model_key
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = _normalize_mode(args.mode)
    train_mode = "opt+r" if mode == "opt+r" else mode
    use_opt_risk = mode == "opt+r"
    train_model = str(spec["train_model"])
    entry_mode = mode
    entry = _ckpt_entry(train_model, entry_mode, dose_fraction, peak)
    metric_path = run_dir / "saved_metric" / dataset_kind / f"{entry}.csv"

    if (not metric_path.exists()) or args.overwrite:
        train_cmd = [
            sys.executable,
            str(repo / "train.py"),
            "--root",
            str(repo),
            "--dataset_kind",
            dataset_kind,
            "--model",
            train_model,
            "--mode",
            train_mode,
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--lr_step_size",
            str(args.lr_step_size),
            "--lr_gamma",
            str(args.lr_gamma),
            "--batch_size",
            str(args.batch_size),
            "--patch_size",
            str(args.patch_size),
            "--num_workers",
            str(args.num_workers),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--dose_fraction",
            str(dose_fraction),
            "--peak",
            str(peak),
        ]

        if train_model in {"unet", "unet_tmp"}:
            train_cmd.extend(
                [
                    "--unet_depth",
                    str(spec["unet_depth"]),
                    "--unet_base_channels",
                    str(spec["unet_base_channels"]),
                ]
            )

        if train_mode == "opt+r":
            train_cmd.extend(
                [
                    "--opt_epochs",
                    str(args.opt_epochs),
                    "--opt_label_root",
                    str(repo / str(cfg["opt_label_root"])),
                    "--opt_burst",
                    str(args.opt_burst),
                ]
            )
            if use_opt_risk:
                train_cmd.append("--opt_use_risk")
                train_cmd.extend(["--opt_risk_input_weight", "0.1"])

        if train_model == "fdk":
            train_cmd.extend(
                [
                    "--fdk_width",
                    str(spec["fdk_width"]),
                    "--fdk_blocks",
                    str(spec["fdk_blocks"]),
                    "--fdk_radius",
                    str(spec["fdk_radius"]),
                ]
            )

        _run(train_cmd, cwd=run_dir)

        metric_cmd = [
            sys.executable,
            str(repo / "calc_metric.py"),
            "--root",
            str(repo),
            "--dataset_kind",
            dataset_kind,
            "--dose_fraction",
            str(dose_fraction),
            "--peak",
            str(peak),
            "--seed",
            str(args.seed),
            "--models",
            entry,
            "--latency_warmup",
            str(args.latency_warmup),
            "--latency_samples",
            str(args.latency_samples),
            "--latency_repeats",
            str(args.latency_repeats),
            "--latency_height",
            str(args.latency_height),
            "--latency_width",
            str(args.latency_width),
        ]
        _run(metric_cmd, cwd=run_dir)

    if not metric_path.exists():
        raise FileNotFoundError(f"Metric file missing: {metric_path}")

    metric = _read_metric(metric_path)

    for key in ("psnr", "ssim", "latency_ms_per_frame"):
        out = run_dir / f"{key}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "model", key])
            w.writeheader()
            w.writerow(
                {"dataset": dataset_name, "model": model_key, key: metric.get(key, "")}
            )

    result_row = {
        "dataset": dataset_name,
        "dataset_kind": dataset_kind,
        "model_key": model_key,
        "train_model": train_model,
        "mode": mode,
        "dose_fraction": _fmt_float(dose_fraction),
        "peak": _fmt_float(peak),
        "entry": entry,
        "fdk_width": str(spec.get("fdk_width", "")),
        "fdk_blocks": str(spec.get("fdk_blocks", "")),
        "fdk_radius": str(spec.get("fdk_radius", "")),
        "fdk_down": str(spec.get("fdk_down", "")),
        "psnr": metric.get("psnr", ""),
        "ssim": metric.get("ssim", ""),
        "latency_ms_per_frame": metric.get("latency_ms_per_frame", ""),
        "metric_csv": str(metric_path),
        "run_dir": str(run_dir),
    }

    result_csv = run_dir / "result.csv"
    with result_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result_row.keys()))
        w.writeheader()
        w.writerow(result_row)

    return result_row


def main() -> None:
    args = _build_parser().parse_args()
    args.mode = _normalize_mode(args.mode)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if args.dose_fraction is not None:
        print(
            "[info] Ignoring --dose_fraction override; using each dataset's data.DEFAULT_LOW_DOSE_CFG['dose_fraction']."
        )

    rows: list[dict[str, str]] = []
    resolved_cfgs: dict[str, dict] = {}
    for dataset_name in args.datasets:
        base_cfg = dict(DATASET_CONFIGS[dataset_name])
        base_cfg["noise_cfg"] = _load_default_noise_cfg(Path(base_cfg["repo"]))
        resolved_cfgs[dataset_name] = base_cfg

    for dataset_name in args.datasets:
        cfg = resolved_cfgs[dataset_name]
        _ensure_opt_label_ready(dataset_name, cfg, args)
        for model_key in args.models:
            rows.append(_run_one(dataset_name, model_key, cfg, args, workspace))

    all_csv = workspace / "all.csv"
    with all_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "dataset_kind",
                "model_key",
                "train_model",
                "mode",
                "dose_fraction",
                "peak",
                "entry",
                "fdk_width",
                "fdk_blocks",
                "fdk_radius",
                "fdk_down",
                "psnr",
                "ssim",
                "latency_ms_per_frame",
                "metric_csv",
                "run_dir",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[done] Saved aggregated results: {all_csv}")


if __name__ == "__main__":
    main()
