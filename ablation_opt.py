from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

CARDIO_MAX_SAMPLES_TOTAL = 2000


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    repo: Path
    dataset_kind: str
    opt_label_root: str


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "weiss": DatasetConfig(
        "weiss", ROOT / "EndovascularSurgery", "weiss", "opt_label_weiss"
    ),
    "jhu": DatasetConfig("jhu", ROOT / "Orthopedic", "jhu", "opt_label_jhu"),
    "cardio": DatasetConfig(
        "cardio", ROOT / "InterventionalCardiology", "cardiac", "opt_label_cardiac"
    ),
}


class CorrAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
        y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
        if x_flat.shape != y_flat.shape:
            raise ValueError(
                f"Correlation shape mismatch: {x_flat.shape} vs {y_flat.shape}"
            )
        if x_flat.size == 0:
            return
        self.n += int(x_flat.size)
        self.sum_x += float(np.sum(x_flat))
        self.sum_y += float(np.sum(y_flat))
        self.sum_x2 += float(np.sum(x_flat * x_flat))
        self.sum_y2 += float(np.sum(y_flat * y_flat))
        self.sum_xy += float(np.sum(x_flat * y_flat))

    def merge(self, other: "CorrAccumulator") -> None:
        self.n += other.n
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_x2 += other.sum_x2
        self.sum_y2 += other.sum_y2
        self.sum_xy += other.sum_xy

    def value(self) -> float:
        if self.n <= 1:
            return float("nan")
        n = float(self.n)
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = n * self.sum_y2 - self.sum_y * self.sum_y
        if den_x <= 0.0 or den_y <= 0.0:
            return float("nan")
        return num / math.sqrt(den_x * den_y)


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = float(torch.mean((x - y) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _fmt_float(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{float(v):.6f}".rstrip("0").rstrip(".")


def _merge_shard_indexes(label_root: Path) -> Path:
    index_path = label_root / "opt_label_index.json"
    if index_path.exists():
        return index_path

    shard_paths = sorted(label_root.glob("opt_label_index.shard*of*.json"))
    if not shard_paths:
        return index_path

    merged_map: Dict[str, Dict[str, int]] = {}
    merged_files: Dict[str, str] = {}
    merged_datasets: Dict[str, str] = {}

    for sp in shard_paths:
        data = json.loads(sp.read_text())
        m = data.get("map", {})
        f = data.get("files", {})
        d = data.get("datasets", {})
        for k, v in m.items():
            merged_map[k] = v
        for k, v in f.items():
            merged_files[k] = v
        for k, v in d.items():
            merged_datasets[k] = v

    index_path.write_text(
        json.dumps(
            {"map": merged_map, "files": merged_files, "datasets": merged_datasets},
            indent=2,
        )
    )
    print(f"[info] merged shard indexes -> {index_path}")
    return index_path


def _ensure_opt_labels(cfg: DatasetConfig) -> None:
    label_root = cfg.repo / cfg.opt_label_root
    label_root.mkdir(parents=True, exist_ok=True)

    index_path = _merge_shard_indexes(label_root)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing opt label index for {cfg.name}: {index_path}. "
            f"Run: python opt_flow_generater.py --datasets {cfg.name}"
        )

    if not any(label_root.glob("*_opt_label.hdf5")):
        raise FileNotFoundError(
            f"Missing opt label hdf5 files for {cfg.name}: {label_root}"
        )


def _build_opt_dataset(cfg: DatasetConfig, data_mod: ModuleType, split: str) -> object:
    return data_mod.DenoiseDataset(
        root=str(cfg.repo),
        split=split,
        patch_size=None,
        mode="opt",
        in_channels=1,
        noise_cfg=dict(data_mod.DEFAULT_LOW_DOSE_CFG),
        dataset_kind=cfg.dataset_kind,
        burst_size=1,
        pretrain=False,
        opt_label_root=str(cfg.repo / cfg.opt_label_root),
        use_opt_risk=True,
    )


def _evaluate_split(
    cfg: DatasetConfig, data_mod: ModuleType, dataset: object, indices: List[int]
) -> Tuple[int, float, float, CorrAccumulator]:
    if not hasattr(dataset, "index"):
        raise RuntimeError("This script expects burst_size=1 and index-based access.")
    corr_acc = CorrAccumulator()
    psnr1_sum = 0.0
    psnr2_sum = 0.0
    used = 0

    with torch.no_grad():
        for i in indices:
            ref = dataset.index[int(i)]
            raw = dataset._load_frame_raw(ref)
            raw = data_mod._select_middle_channel(raw)
            if raw.dim() == 2:
                raw = raw.unsqueeze(0)
            bbox = data_mod.compute_black_border_bbox(raw)

            clean = dataset._load_frame(ref, bbox=bbox)
            noisy = data_mod.corrupt(
                clean.unsqueeze(0).clone(), dataset.noise_cfg
            ).squeeze(0)
            label = dataset._load_opt_label(ref, bbox=bbox)
            unc = dataset._load_opt_risk(ref, bbox=bbox)

            psnr1_sum += _psnr(noisy, clean)
            psnr2_sum += _psnr(label, clean)

            abs_err = torch.abs(clean - label)
            corr_acc.update(unc.cpu().numpy(), abs_err.cpu().numpy())
            used += 1

    return used, psnr1_sum, psnr2_sum, corr_acc


def _evaluate_dataset(
    cfg: DatasetConfig, max_samples_per_split: int | None, seed: int
) -> Tuple[float, float, float, int]:
    data_mod = _load_module(cfg.repo / "data.py", f"ablation_new_opt_data_{cfg.name}")
    train_ds = _build_opt_dataset(cfg, data_mod, "training")
    val_ds = _build_opt_dataset(cfg, data_mod, "validation")

    train_indices = list(range(len(train_ds)))
    val_indices = list(range(len(val_ds)))

    if max_samples_per_split is not None:
        k = max(0, int(max_samples_per_split))
        train_indices = train_indices[:k]
        val_indices = val_indices[:k]
    elif cfg.name == "cardio":
        total_all = len(train_indices) + len(val_indices)
        sample_n = min(int(CARDIO_MAX_SAMPLES_TOTAL), int(total_all))
        if sample_n < total_all:
            rng = np.random.default_rng(int(seed))
            picked = rng.choice(total_all, size=sample_n, replace=False)
            train_pick = sorted(int(x) for x in picked if int(x) < len(train_indices))
            val_pick = sorted(
                int(x) - len(train_indices)
                for x in picked
                if int(x) >= len(train_indices)
            )
            train_indices = train_pick
            val_indices = val_pick
        print(
            f"[info] cardio sampling: total_trainval={total_all}, sampled={len(train_indices) + len(val_indices)}"
        )

    total_count = 0
    psnr1_sum = 0.0
    psnr2_sum = 0.0
    corr_acc = CorrAccumulator()

    for split, ds, idxs in [
        ("training", train_ds, train_indices),
        ("validation", val_ds, val_indices),
    ]:
        n, p1, p2, split_corr = _evaluate_split(cfg, data_mod, ds, idxs)
        print(f"[info] {cfg.name}/{split}: {n} samples")
        total_count += n
        psnr1_sum += p1
        psnr2_sum += p2
        corr_acc.merge(split_corr)

    if total_count <= 0:
        raise RuntimeError(f"No train/val samples available for dataset: {cfg.name}")

    return (
        psnr1_sum / total_count,
        psnr2_sum / total_count,
        corr_acc.value(),
        total_count,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation for new opt labels (final).")
    p.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASET_CONFIGS),
        default=["weiss", "jhu", "cardio"],
    )
    p.add_argument("--output", default="newopt_abla.csv")
    p.add_argument("--max_samples_per_split", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    rows: List[Dict[str, str]] = []
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIGS[str(dataset_name)]
        print(f"\n[info] dataset={cfg.name}")
        _ensure_opt_labels(cfg)
        psnr1, psnr2, corr, n = _evaluate_dataset(
            cfg, args.max_samples_per_split, seed=int(args.seed)
        )
        print(
            f"[result] {cfg.name}: n={n}, "
            f"psnr1={_fmt_float(psnr1)}, psnr2={_fmt_float(psnr2)}, corr={_fmt_float(corr)}"
        )
        rows.append(
            {
                "dataset": cfg.name,
                "corrupted": _fmt_float(psnr1),
                "opt+r": _fmt_float(psnr2),
                "opt+r-risk": _fmt_float(corr),
            }
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "corrupted", "opt+r", "opt+r-risk"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[done] Saved report: {out_path}")


if __name__ == "__main__":
    main()
