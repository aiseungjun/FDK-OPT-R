from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
from calc_metric import _infer_dataset_tag, load_model_entry
from data import DenoiseDataset


def _fmt_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _sobel_kernels(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return kx, ky


def _gradient_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    c = pred.shape[1]
    kx, ky = _sobel_kernels(pred.device, pred.dtype)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    pred_gx = F.conv2d(pred, kx, padding=1, groups=c)
    pred_gy = F.conv2d(pred, ky, padding=1, groups=c)
    tgt_gx = F.conv2d(target, kx, padding=1, groups=c)
    tgt_gy = F.conv2d(target, ky, padding=1, groups=c)

    return float(
        torch.sqrt(F.mse_loss(pred_gx, tgt_gx) + F.mse_loss(pred_gy, tgt_gy)).item()
    )


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
def _evaluate_model_edges(
    model: torch.nn.Module,
    *,
    root: Path,
    dataset_tag: str,
    in_channels: int,
    burst_size: int,
    eval_opts: dict,
    noise_cfg: dict,
    device: torch.device,
) -> float:
    ds = DenoiseDataset(
        root=root,
        split="test",
        patch_size=None,
        mode="n2c",
        in_channels=in_channels,
        noise_cfg=noise_cfg,
        dataset_kind=dataset_tag,
        burst_size=burst_size,
        burst_align=bool(eval_opts.get("burst_align", False)),
        burst_align_max_shift=float(eval_opts.get("burst_align_max_shift", 10.0)),
        burst_causal=bool(eval_opts.get("burst_causal", False)),
        burst_target=str(eval_opts.get("burst_target", "center")),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    target_pos = eval_opts.get("target_pos")

    values: list[float] = []
    for inp, target in loader:
        inp = inp.to(device)
        target = target.to(device)
        target_frame = _select_target_if_burst(target, burst_size, target_pos)
        out = _run_model(model, inp, noise_cfg)
        out = _select_target_if_burst(out, burst_size, target_pos)
        values.append(_gradient_rmse(out, target_frame))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset_kind", default="weiss", choices=["weiss"])
    parser.add_argument("--out", default="edge_results.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
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

    print(f"[Edge eval] dataset={dataset_tag} root={root} device={device}")
    rows: list[dict[str, str]] = []

    for display_name, entry_name in methods:
        print(f"[Edge eval] {display_name} ({entry_name})")
        try:
            model, _mode, noise_cfg, in_channels, burst_size, _lookup, eval_opts = (
                load_model_entry(
                    entry_name,
                    device,
                    dataset_tag,
                    dose_fraction,
                    peak,
                )
            )
            rmse = _evaluate_model_edges(
                model,
                root=root,
                dataset_tag=dataset_tag,
                in_channels=in_channels,
                burst_size=burst_size,
                eval_opts=eval_opts,
                noise_cfg=noise_cfg,
                device=device,
            )
            print(f"  -> gradient_rmse={rmse:.6f}")
            rows.append({"model": display_name, "gradient_rmse": f"{rmse:.6f}"})
        except FileNotFoundError as exc:
            print(f"  -> [skip] checkpoint missing: {exc}")
        except Exception as exc:
            print(f"  -> [error] {exc}")

    out_path = Path(args.out)
    if rows:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "gradient_rmse"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] saved: {out_path.resolve()}")
    else:
        print("[warn] no results to save")


if __name__ == "__main__":
    main()
