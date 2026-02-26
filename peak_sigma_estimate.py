from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Job:
    name: str
    module_path: Path
    root: Path
    dataset_kind: str
    sample_frames: int | None


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flatten_groups(
    groups: dict[Any, list[tuple[int, Any]]],
) -> list[tuple[str, int, Any]]:
    refs: list[tuple[str, int, Any]] = []
    for group_id, items in groups.items():
        for frame_idx, ref in items:
            refs.append((str(group_id), int(frame_idx), ref))
    refs.sort(key=lambda x: (x[0], x[1]))
    return refs


def _random_sample(
    rows: list[tuple[str, int, Any]], limit: int | None
) -> list[tuple[str, int, Any]]:
    if not rows:
        return []
    if limit is None or len(rows) <= int(limit):
        out = list(rows)
        random.shuffle(out)
        return out
    n = max(1, int(limit))
    return random.sample(rows, k=n)


def _group_range(rows: list[tuple[str, int, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for group_id, frame_idx, _ref in rows:
        bucket = out.setdefault(
            group_id, {"min": frame_idx, "max": frame_idx, "count": 0}
        )
        bucket["min"] = min(bucket["min"], frame_idx)
        bucket["max"] = max(bucket["max"], frame_idx)
        bucket["count"] += 1
    return out


def _estimator_defaults(mod) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    sig = inspect.signature(mod.estimate_poisson_gaussian_params)
    for name, param in sig.parameters.items():
        if name == "x_disp":
            continue
        if param.default is not inspect._empty:
            defaults[name] = param.default
    return defaults


def _run_job(job: Job, *, full_bbox_value: int) -> dict[str, Any]:
    module = _load_module(job.module_path.resolve(), f"{job.name}_data_mod")
    dataset = module.DenoiseDataset(
        root=job.root,
        split="training",
        patch_size=None,
        mode="n2c",
        in_channels=1,
        noise_cfg=None,
        dataset_kind=job.dataset_kind,
        burst_size=1,
        pretrain=False,
    )

    groups = dataset.frame_groups
    all_refs = _flatten_groups(groups)
    used_refs = _random_sample(all_refs, job.sample_frames)

    full_bbox = (0, int(full_bbox_value), 0, int(full_bbox_value))
    frames = [
        dataset._load_frame(ref, bbox=full_bbox) for _group, _idx, ref in used_refs
    ]
    x = torch.stack(frames, dim=0).to(torch.float32)

    peak, sigma_read = module.estimate_poisson_gaussian_params(x)

    result = {
        "name": job.name,
        "dataset_kind": job.dataset_kind,
        "module_path": str(job.module_path),
        "root": str(job.root),
        "total_training_groups": len(groups),
        "total_training_frames": len(all_refs),
        "used_training_frames": len(used_refs),
        "sampling": "random_all"
        if job.sample_frames is None
        else f"random_{int(job.sample_frames)}",
        "preprocess_for_estimation": (
            f"DenoiseDataset._load_frame(ref, bbox=(0,{int(full_bbox_value)},0,{int(full_bbox_value)}))"
        ),
        "stack_shape": list(x.shape),
        "estimator_defaults": _estimator_defaults(module),
        "estimate_peak": float(peak),
        "estimate_sigma_read": float(sigma_read),
        "used_first_ref": {"group": used_refs[0][0], "frame_idx": used_refs[0][1]}
        if used_refs
        else None,
        "used_last_ref": {"group": used_refs[-1][0], "frame_idx": used_refs[-1][1]}
        if used_refs
        else None,
        "all_training_group_ranges": _group_range(all_refs),
        "used_group_ranges": _group_range(used_refs),
    }

    closer = getattr(dataset, "close", None)
    if callable(closer):
        closer()
    h5_store = getattr(dataset, "h5_store", None)
    if h5_store is not None and hasattr(h5_store, "close"):
        h5_store.close()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate peak/sigma_read from training frames."
    )
    parser.add_argument(
        "--output",
        default="noise_param_estimates_random.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--interventional_frames",
        type=int,
        default=512,
        help="Frame count for InterventionalCardiology random sampling (default: 512).",
    )
    parser.add_argument(
        "--full_bbox_value",
        type=int,
        default=100000,
        help="Large bbox edge value used to effectively disable per-frame border cropping.",
    )
    args = parser.parse_args()

    jobs = [
        Job(
            name="EndovascularSurgery",
            module_path=Path("EndovascularSurgery/data.py"),
            root=Path("EndovascularSurgery"),
            dataset_kind="weiss",
            sample_frames=None,
        ),
        Job(
            name="Orthopedic",
            module_path=Path("Orthopedic/data.py"),
            root=Path("Orthopedic"),
            dataset_kind="jhu",
            sample_frames=None,
        ),
        Job(
            name="InterventionalCardiology",
            module_path=Path("InterventionalCardiology/data.py"),
            root=Path("InterventionalCardiology"),
            dataset_kind="cardiac",
            sample_frames=int(args.interventional_frames),
        ),
    ]

    results: list[dict[str, Any]] = []
    for job in jobs:
        result = _run_job(job, full_bbox_value=args.full_bbox_value)
        results.append(result)
        print(f"=== {job.name} ===")
        print(f"dataset_kind={result['dataset_kind']}")
        print(
            f"training_groups={result['total_training_groups']}, "
            f"training_frames={result['total_training_frames']}"
        )
        print(
            f"used_frames={result['used_training_frames']} (sampling={result['sampling']})"
        )
        print(f"stack_shape={tuple(result['stack_shape'])}")
        print(f"estimate_peak={result['estimate_peak']:.6f}")
        print(f"estimate_sigma_read={result['estimate_sigma_read']:.6f}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_mode": "random_no_fixed_seed",
        "results": results,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
