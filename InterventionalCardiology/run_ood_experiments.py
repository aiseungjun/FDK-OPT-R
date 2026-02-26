from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import data
from calc_metric import _infer_dataset_tag, load_model_entry, psnr_torch, ssim_torch
from data_ood import OODDenoiseDataset, OODSpec

DEFAULT_TEST_SAMPLE_COUNT = 500
DEFAULT_TEST_SAMPLE_SEED = 2026


def _fmt_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _select_target_if_burst(
    x: torch.Tensor, burst_size: int, target_pos: int | None
) -> torch.Tensor:
    if x.dim() == 5:
        pos = burst_size // 2 if target_pos is None else int(target_pos)
        pos = max(0, min(pos, x.size(1) - 1))
        return x[:, pos]
    return x


def _run_model(
    model: torch.nn.Module, inp: torch.Tensor, noise_cfg: dict
) -> torch.Tensor:
    from model import FFDNet, FastDVDnet

    if isinstance(model, (FastDVDnet, FFDNet)):
        noise_level = data.get_noise_level(noise_cfg)
        h, w = inp.size(-2), inp.size(-1)
        noise_map = torch.full(
            (inp.size(0), 1, h, w),
            float(noise_level),
            device=inp.device,
            dtype=inp.dtype,
        )
        return model(inp, noise_map)
    return model(inp)


@torch.inference_mode()
def _eval_ood_condition(
    model: torch.nn.Module,
    *,
    root: Path,
    dataset_tag: str,
    in_channels: int,
    burst_size: int,
    eval_opts: dict,
    noise_cfg: dict,
    ood_spec: OODSpec,
    device: torch.device,
    seed: int,
    sample_indices: list[int] | None = None,
) -> tuple[float, float]:
    _seed_all(seed)

    ds = OODDenoiseDataset(
        root,
        "test",
        None,
        "n2c",
        in_channels,
        noise_cfg,
        dataset_kind=dataset_tag,
        burst_size=burst_size,
        burst_align=bool(eval_opts.get("burst_align", False)),
        burst_align_max_shift=float(eval_opts.get("burst_align_max_shift", 10.0)),
        burst_causal=bool(eval_opts.get("burst_causal", False)),
        burst_target=str(eval_opts.get("burst_target", "center")),
        ood_spec=ood_spec,
    )

    if sample_indices:
        max_idx = int(max(sample_indices))
        if max_idx >= len(ds):
            raise IndexError(
                f"Sample index out of range for OOD test set: max_idx={max_idx}, len={len(ds)}"
            )
        ds = Subset(ds, list(sample_indices))

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    target_pos = eval_opts.get("target_pos")

    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    model.eval()

    for inp, target in loader:
        inp = inp.to(device)
        target = target.to(device)

        target_frame = _select_target_if_burst(target, burst_size, target_pos)
        out = _run_model(model, inp, noise_cfg)
        out = _select_target_if_burst(out, burst_size, target_pos)
        out = torch.clamp(out, 0.0, 1.0)

        psnr_vals.append(float(psnr_torch(out, target_frame).item()))
        ssim_vals.append(float(ssim_torch(out, target_frame).item()))

    psnr_mean = float(sum(psnr_vals) / max(1, len(psnr_vals)))
    ssim_mean = float(sum(ssim_vals) / max(1, len(ssim_vals)))
    return psnr_mean, ssim_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=None, help="Dataset root (defaults to this script folder)"
    )
    parser.add_argument("--dataset_kind", default="cardiac", choices=["cardiac"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--test_max_samples", type=int, default=DEFAULT_TEST_SAMPLE_COUNT
    )
    parser.add_argument(
        "--test_sample_seed", type=int, default=DEFAULT_TEST_SAMPLE_SEED
    )
    parser.add_argument("--out", default="ood_results.csv")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root is not None else script_dir
    dataset_tag = _infer_dataset_tag(root, args.dataset_kind)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dose_fraction = float(data.DEFAULT_LOW_DOSE_CFG["dose_fraction"])
    peak = float(data.DEFAULT_LOW_DOSE_CFG["peak"])
    dose_tag = _fmt_float(dose_fraction)
    peak_tag = _fmt_float(peak)

    methods: List[Tuple[str, str]] = [
        ("FDK-OPT+R", f"fdk-opt+r-{dose_tag}-{peak_tag}"),
        ("UNet-N2V", f"unet-n2v-{dose_tag}-{peak_tag}"),
        ("UNet-R2R", f"unet-r2r-{dose_tag}-{peak_tag}"),
        ("NAFNet", f"nafnet-n2c-{dose_tag}-{peak_tag}"),
        ("UNet-N2C", f"unet-n2c-{dose_tag}-{peak_tag}"),
        ("UNet-OPT+R", f"unet-opt+r-{dose_tag}-{peak_tag}"),
    ]

    ood_grid: List[Tuple[str, List[float]]] = [
        ("dose_shift", [0.3, 0.4]),
        ("gain", [0.9, 1.1]),
        ("scatter", [0.05, 0.1]),
    ]

    columns: List[str] = ["method"]
    for ood_kind, xs in ood_grid:
        for x in xs:
            tag = f"{ood_kind}@{_fmt_float(x)}"
            columns.append(f"{tag}_psnr")
            columns.append(f"{tag}_ssim")

    print(f"[OOD eval] dataset={dataset_tag} root={root} device={device}")
    print(f"[OOD eval] writing -> {args.out}")

    sample_indices: list[int] | None = None
    test_max_samples = int(args.test_max_samples)
    if test_max_samples > 0:
        groups = data.get_frame_groups(
            root=root, split="test", dataset_kind=dataset_tag, pretrain=False
        )
        total_test = sum(len(items) for items in groups.values())
        if total_test <= 0:
            raise RuntimeError(
                f"No test frames found for dataset={dataset_tag} under root={root}"
            )
        sample_n = min(test_max_samples, int(total_test))
        rng = random.Random(int(args.test_sample_seed))
        sample_indices = sorted(rng.sample(range(total_test), sample_n))
        print(
            f"[OOD eval] using fixed test subset: {sample_n}/{total_test} (seed={int(args.test_sample_seed)})"
        )
    else:
        print("[OOD eval] using full test set (no sampling)")

    rows: list[Dict[str, str]] = []
    for method_name, entry in methods:
        print("\n" + "=" * 80)
        print(f"[model] {method_name} (entry={entry})")

        try:
            model, _mode, noise_cfg, in_channels, burst_size, _lookup, eval_opts = (
                load_model_entry(
                    entry,
                    device,
                    dataset_tag,
                    dose_fraction,
                    peak,
                )
            )
        except FileNotFoundError as exc:
            print(f"  -> [skip] checkpoint missing: {exc}")
            continue

        row: Dict[str, str] = {"method": method_name}
        for ood_kind, xs in ood_grid:
            for x in xs:
                tag = f"{ood_kind}@{_fmt_float(x)}"
                ood_spec = OODSpec(kind=ood_kind, x=float(x), scatter_sigma=7.0)
                psnr, ssim = _eval_ood_condition(
                    model,
                    root=root,
                    dataset_tag=dataset_tag,
                    in_channels=in_channels,
                    burst_size=burst_size,
                    eval_opts=eval_opts,
                    noise_cfg=noise_cfg,
                    ood_spec=ood_spec,
                    device=device,
                    seed=int(args.seed),
                    sample_indices=sample_indices,
                )
                row[f"{tag}_psnr"] = f"{psnr:.6f}"
                row[f"{tag}_ssim"] = f"{ssim:.6f}"
                print(f"  - {tag:<16} PSNR={psnr:.4f} SSIM={ssim:.4f}")
        rows.append(row)

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[done] saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
