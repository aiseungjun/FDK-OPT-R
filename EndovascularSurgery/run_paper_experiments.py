import argparse
import csv
import subprocess
import sys
from pathlib import Path

from data import DEFAULT_LOW_DOSE_CFG

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_SEEDS = [1337, 6324, 2346, 7352, 2734]


def _fmt_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _normalize_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"opt", "optr", "opt+r"}:
        return "opt+r"
    return mode


def _entry(model: str, mode: str, dose: float, peak: float) -> str:
    mode_tag = _normalize_mode(mode)
    return f"{model}-{mode_tag}-{_fmt_float(dose)}-{_fmt_float(peak)}"


def _train_one(
    dataset_kind: str,
    model: str,
    mode: str,
    *,
    root: Path,
    epochs: int,
    opt_epochs: int,
    lr: float,
    batch_size: int,
    patch_size: int,
    device: str,
    num_workers: int,
    seed: int,
    opt_label_root: str,
    opt_use_risk: bool = False,
    lr_step_size: int = 10,
    lr_gamma: float = 0.2,
    fdk_width: int = 32,
    fdk_blocks: int = 4,
    fdk_radius: int = 1,
    extra: list[str] | None = None,
) -> None:
    mode_tag = _normalize_mode(mode)
    train_mode = "opt" if mode_tag == "opt+r" else mode_tag
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train.py"),
        "--root",
        str(root),
        "--dataset_kind",
        dataset_kind,
        "--model",
        model,
        "--mode",
        train_mode,
        "--epochs",
        str(epochs),
        "--opt_epochs",
        str(opt_epochs),
        "--lr",
        str(lr),
        "--lr_step_size",
        str(lr_step_size),
        "--lr_gamma",
        str(lr_gamma),
        "--batch_size",
        str(batch_size),
        "--patch_size",
        str(patch_size),
        "--num_workers",
        str(num_workers),
        "--device",
        device,
        "--seed",
        str(seed),
        "--opt_label_root",
        opt_label_root,
        "--opt_burst",
        "5",
    ]
    if model == "fdk":
        cmd.extend(
            [
                "--fdk_width",
                str(fdk_width),
                "--fdk_blocks",
                str(fdk_blocks),
                "--fdk_radius",
                str(fdk_radius),
            ]
        )
    if mode_tag == "opt+r" or opt_use_risk:
        cmd.append("--opt_use_risk")
    if extra:
        cmd.extend(extra)
    _run(cmd)


def _build_paper_table(
    dataset: str, dose: float, peak: float, metric_path: Path, out_path: Path
) -> None:
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)

    rows = []
    with metric_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    dose_tag = _fmt_float(dose)
    peak_tag = _fmt_float(peak)

    preferred = [
        _entry("unet", "opt+r", dose, peak),
        _entry("fdk", "opt+r", dose, peak),
        _entry("bm3d", "n2c", dose, peak),
        _entry("nlm", "n2c", dose, peak),
        _entry("unet", "n2v", dose, peak),
        _entry("unet", "n2self", dose, peak),
        _entry("unet", "r2r", dose, peak),
        _entry("unet", "n2c", dose, peak),
        _entry("redcnn", "n2c", dose, peak),
        _entry("wganvgg", "n2c", dose, peak),
        _entry("dncnn", "n2c", dose, peak),
        _entry("ffdnet", "n2c", dose, peak),
        _entry("nafnet", "n2c", dose, peak),
        _entry("fastdvdnet", "n2c", dose, peak),
        _entry("edvr", "n2c", dose, peak),
    ]

    row_map = {r["entry"]: r for r in rows}
    out_rows = []
    for key in preferred:
        if key not in row_map:
            continue
        r = row_map[key]
        out_rows.append(
            {
                "dataset": dataset,
                "entry": key,
                "model": r.get("model", ""),
                "mode": r.get("mode", ""),
                "psnr": r.get("psnr", ""),
                "ssim": r.get("ssim", ""),
                "latency_ms_per_frame": r.get("latency_ms_per_frame", ""),
                "dose_fraction": dose_tag,
                "peak": peak_tag,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "entry",
                "model",
                "mode",
                "psnr",
                "ssim",
                "latency_ms_per_frame",
                "dose_fraction",
                "peak",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)


def _resolve_run_seeds(num_ex: int, base_seed: int) -> list[int]:
    if int(num_ex) < 1:
        raise ValueError("num_ex must be >= 1")
    if int(num_ex) == 1:
        return [int(base_seed)]
    if int(num_ex) > len(EXPERIMENT_SEEDS):
        raise ValueError(
            f"num_ex={int(num_ex)} exceeds predefined EXPERIMENT_SEEDS length={len(EXPERIMENT_SEEDS)}. "
            "Add more seeds to EXPERIMENT_SEEDS if needed."
        )
    return [int(s) for s in EXPERIMENT_SEEDS[: int(num_ex)]]


def _require_peak_sigma_config() -> tuple[float, float]:
    peak = DEFAULT_LOW_DOSE_CFG.get("peak", None)
    sigma_read = DEFAULT_LOW_DOSE_CFG.get("sigma_read", None)
    if peak is None or sigma_read is None:
        raise ValueError(
            "DEFAULT_LOW_DOSE_CFG['peak'] and ['sigma_read'] are not set. "
            "Run `python peak_sigma_estimate.py` in `paper_code/`, then copy "
            "`estimate_peak` and `estimate_sigma_read` into EndovascularSurgery/data.py."
        )
    return float(peak), float(sigma_read)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end WEISS paper experiments."
    )
    parser.add_argument("--root", default=str(SCRIPT_DIR))
    parser.add_argument("--dataset", default="weiss", choices=["weiss"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--opt_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_ex", type=int, default=1)
    parser.add_argument("--skip_opt_label", action="store_true")
    parser.add_argument("--fdk_width", type=int, default=32)
    parser.add_argument("--fdk_blocks", type=int, default=4)
    parser.add_argument("--fdk_radius", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.root)
    datasets = ["weiss"]
    run_seeds = _resolve_run_seeds(args.num_ex, args.seed)
    print(f"[info] run seeds: {run_seeds}")

    for dataset in datasets:
        dose = float(DEFAULT_LOW_DOSE_CFG["dose_fraction"])
        peak, _sigma_read = _require_peak_sigma_config()
        opt_label_root = f"opt_label_{dataset}"
        fdk_epochs = int(args.opt_epochs)

        if not args.skip_opt_label:
            paper_root = root.parent
            _run(
                [
                    sys.executable,
                    str(paper_root / "opt_flow_generater.py"),
                    "--root",
                    str(paper_root),
                    "--datasets",
                    dataset,
                ]
            )

        for run_idx, run_seed in enumerate(run_seeds, start=1):
            print(f"[info] [{run_idx}/{len(run_seeds)}] running with seed={run_seed}")

            _train_one(
                dataset,
                "unet",
                "opt+r",
                root=root,
                epochs=args.epochs,
                opt_epochs=args.opt_epochs,
                lr=args.lr,
                lr_step_size=args.lr_step_size,
                lr_gamma=args.lr_gamma,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                device=args.device,
                num_workers=args.num_workers,
                seed=run_seed,
                opt_label_root=opt_label_root,
            )
            _train_one(
                dataset,
                "fdk",
                "opt+r",
                root=root,
                epochs=fdk_epochs,
                opt_epochs=fdk_epochs,
                lr=args.lr,
                lr_step_size=args.lr_step_size,
                lr_gamma=args.lr_gamma,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                device=args.device,
                num_workers=args.num_workers,
                seed=run_seed,
                opt_label_root=opt_label_root,
                fdk_width=args.fdk_width,
                fdk_blocks=args.fdk_blocks,
                fdk_radius=args.fdk_radius,
            )

            for mode in ["n2v", "n2self", "r2r"]:
                _train_one(
                    dataset,
                    "unet",
                    mode,
                    root=root,
                    epochs=args.epochs,
                    opt_epochs=args.opt_epochs,
                    lr=args.lr,
                    lr_step_size=args.lr_step_size,
                    lr_gamma=args.lr_gamma,
                    batch_size=args.batch_size,
                    patch_size=args.patch_size,
                    device=args.device,
                    num_workers=args.num_workers,
                    seed=run_seed,
                    opt_label_root=opt_label_root,
                )

            for model in [
                "unet",
                "redcnn",
                "wganvgg",
                "dncnn",
                "ffdnet",
                "nafnet",
                "fastdvdnet",
                "edvr",
            ]:
                _train_one(
                    dataset,
                    model,
                    "n2c",
                    root=root,
                    epochs=args.epochs,
                    opt_epochs=args.opt_epochs,
                    lr=args.lr,
                    lr_step_size=args.lr_step_size,
                    lr_gamma=args.lr_gamma,
                    batch_size=args.batch_size,
                    patch_size=args.patch_size,
                    device=args.device,
                    num_workers=args.num_workers,
                    seed=run_seed,
                    opt_label_root=opt_label_root,
                )

            metric_entries = [
                _entry("unet", "opt+r", dose, peak),
                _entry("fdk", "opt+r", dose, peak),
                _entry("bm3d", "n2c", dose, peak),
                _entry("nlm", "n2c", dose, peak),
                _entry("unet", "n2v", dose, peak),
                _entry("unet", "n2self", dose, peak),
                _entry("unet", "r2r", dose, peak),
                _entry("unet", "n2c", dose, peak),
                _entry("redcnn", "n2c", dose, peak),
                _entry("wganvgg", "n2c", dose, peak),
                _entry("dncnn", "n2c", dose, peak),
                _entry("ffdnet", "n2c", dose, peak),
                _entry("nafnet", "n2c", dose, peak),
                _entry("fastdvdnet", "n2c", dose, peak),
                _entry("edvr", "n2c", dose, peak),
            ]

            _run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "calc_metric.py"),
                    "--root",
                    str(root),
                    "--dataset_kind",
                    dataset,
                    "--seed",
                    str(run_seed),
                    "--models",
                    ",".join(metric_entries),
                    "--latency_warmup",
                    "30",
                    "--latency_samples",
                    "200",
                    "--latency_repeats",
                    "5",
                    "--latency_height",
                    "256",
                    "--latency_width",
                    "256",
                ]
            )

            metric_summary = Path("saved_metric") / dataset / f"summary_{dataset}.csv"
            if len(run_seeds) == 1:
                paper_out = (
                    Path("saved_metric") / dataset / f"paper_comparison_{dataset}.csv"
                )
            else:
                paper_out = (
                    Path("saved_metric")
                    / dataset
                    / f"paper_comparison_{dataset}{run_idx}.csv"
                )
            _build_paper_table(dataset, dose, peak, metric_summary, paper_out)
            print(f"Saved paper table: {paper_out}")


if __name__ == "__main__":
    main()
