import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from data import DEFAULT_LOW_DOSE_CFG, get_dataloaders, get_noise_level
from model import (
    BM3D,
    DnCNN,
    EDVR,
    FFDNet,
    FastDVDnet,
    FDK,
    NAFNet,
    NLM,
    REDCNN,
    UNet,
    UNetTemp,
    WGANVGGDiscriminator,
    WGANVGGGenerator,
)

OPT_RISK_INPUT_WEIGHT = 0.1


def _fmt_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _normalize_mode_alias(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"opt+r"}:
        return "opt+r"
    return mode


def _is_opt_mode(mode: str) -> bool:
    return _normalize_mode_alias(mode) == "opt+r"


def _mode_tag(mode: str, use_opt_risk: bool) -> str:
    if _is_opt_mode(mode):
        return "opt+r"
    return str(mode).lower()


def _use_perceptual(mode: str, perceptual_weight: float) -> bool:
    mode = _normalize_mode_alias(mode)
    return mode in {"pretrain", "opt+r"} and perceptual_weight > 0.0


class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except Exception as exc:
            raise RuntimeError(
                "torchvision is required for VGG perceptual loss. "
                "Install torchvision or set --perceptual_weight 0."
            ) from exc

        try:
            features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception as exc:
            print(
                f"[warn] Failed to load pretrained VGG16 weights ({exc}). Falling back to random VGG16."
            )
            features = vgg16(weights=None).features

        self.blocks = nn.ModuleList(
            [
                features[:4].eval(),
                features[4:9].eval(),
                features[9:16].eval(),
                features[16:23].eval(),
            ]
        )
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @staticmethod
    def _to_rgb(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
        channels = x.size(1)
        if channels == 3:
            return x
        if channels == 1:
            return x.repeat(1, 3, 1, 1)
        if channels > 3:
            return x[:, :3]
        return torch.cat([x, x[:, :1]], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rgb = self._to_rgb(torch.clamp(pred, 0.0, 1.0))
        target_rgb = self._to_rgb(torch.clamp(target, 0.0, 1.0))
        pred_norm = (pred_rgb - self.mean) / self.std
        target_norm = (target_rgb - self.mean) / self.std

        loss = pred_norm.new_tensor(0.0)
        x = pred_norm
        y = target_norm
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return loss / float(len(self.blocks))


def build_model(name: str, in_channels: int = 1, **kwargs):
    name = name.lower()
    if name == "unet":
        return UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=int(kwargs.get("base_channels", 32)),
            depth=int(kwargs.get("depth", 3)),
        )
    if name in {"unet_tmp", "unettmp", "unettemp", "unet-temp"}:
        return UNetTemp(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=int(kwargs.get("base_channels", 32)),
            depth=int(kwargs.get("depth", 3)),
            temporal_attn_temp=float(kwargs.get("temporal_attn_temp", 0.1)),
            temporal_input_fusion=bool(kwargs.get("temporal_input_fusion", True)),
            temporal_center_only=bool(kwargs.get("temporal_center_only", True)),
        )
    if name == "fastdvdnet":
        return FastDVDnet(
            in_channels=in_channels,
            num_frames=int(kwargs.get("num_frames", 5)),
            default_noise_level=float(kwargs.get("default_noise_level", 0.03)),
            base_channels=int(kwargs.get("base_channels", 32)),
        )
    if name == "fdk":
        return FDK(in_channels=in_channels, out_channels=in_channels, **kwargs)
    if name == "dncnn":
        return DnCNN(
            in_channels=in_channels,
            depth=int(kwargs.get("depth", 17)),
            features=int(kwargs.get("features", 64)),
        )
    if name in {"redcnn", "red-cnn"}:
        return REDCNN(
            in_channels=in_channels,
            channels=int(kwargs.get("channels", 96)),
            num_layers=int(kwargs.get("num_layers", 5)),
        )
    if name in {"wganvgg", "wgan-vgg"}:
        return WGANVGGGenerator(
            in_channels=in_channels,
            base_channels=int(kwargs.get("base_channels", 32)),
            num_blocks=int(kwargs.get("num_blocks", 8)),
        )
    if name == "ffdnet":
        return FFDNet(
            in_channels=in_channels,
            features=int(kwargs.get("features", 64)),
            depth=int(kwargs.get("depth", 15)),
            default_noise_level=float(kwargs.get("default_noise_level", 0.03)),
        )
    if name == "nafnet":
        return NAFNet(
            in_channels=in_channels,
            width=int(kwargs.get("width", 32)),
            blocks=int(kwargs.get("blocks", 6)),
        )
    if name == "edvr":
        return EDVR(
            in_channels=in_channels,
            num_frames=int(kwargs.get("num_frames", 5)),
            channels=int(kwargs.get("channels", 32)),
            num_blocks=int(kwargs.get("num_blocks", 10)),
        )
    if name == "bm3d":
        return BM3D()
    if name == "nlm":
        return NLM()
    raise ValueError(f"Unknown model: {name}")


def unpack_batch(batch, mode: str, use_risk: bool = False):
    if mode in {"n2c", "r2r", "pretrain"}:
        inp, target = batch
        mask = None
        unc = None
    elif _is_opt_mode(mode):
        if use_risk:
            inp, target, unc = batch
        else:
            inp, target = batch
            unc = None
        mask = None
    else:
        inp, target, mask = batch
        unc = None
    return inp, target, mask, unc


def _select_center_if_burst(
    x: torch.Tensor, burst_size: int, target_pos: int | None = None
) -> torch.Tensor:
    if x.dim() == 5:
        pos = burst_size // 2 if target_pos is None else int(target_pos)
        pos = max(0, min(pos, x.size(1) - 1))
        return x[:, pos]
    return x


def _ssim_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c = output.size(1)
    window_size = 11
    sigma = 1.5
    coords = (
        torch.arange(window_size, device=output.device, dtype=output.dtype)
        - (window_size - 1) / 2
    )
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (
        (g[:, None] * g[None, :])
        .view(1, 1, window_size, window_size)
        .repeat(c, 1, 1, 1)
    )

    c1 = 0.01**2
    c2 = 0.03**2
    mu1 = torch.nn.functional.conv2d(output, window, padding=window_size // 2, groups=c)
    mu2 = torch.nn.functional.conv2d(target, window, padding=window_size // 2, groups=c)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2
    sigma1_sq = (
        torch.nn.functional.conv2d(
            output * output, window, padding=window_size // 2, groups=c
        )
        - mu1_sq
    )
    sigma2_sq = (
        torch.nn.functional.conv2d(
            target * target, window, padding=window_size // 2, groups=c
        )
        - mu2_sq
    )
    sigma12 = (
        torch.nn.functional.conv2d(
            output * target, window, padding=window_size // 2, groups=c
        )
        - mu12
    )
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return 1.0 - ssim_map.mean()


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    burst_size: int,
    *,
    target_pos: int | None = None,
    risk: torch.Tensor | None = None,
    noisy_input: torch.Tensor | None = None,
    unc_eps: float = 1e-3,
    unc_power: float = 1.0,
    unc_max_weight: float = 4.0,
) -> torch.Tensor:
    output = _select_center_if_burst(output, burst_size, target_pos)
    target_center = _select_center_if_burst(target, burst_size, target_pos)
    if mask is None:
        if risk is None:
            return nn.functional.mse_loss(output, target_center)
        unc = _select_center_if_burst(risk, burst_size, target_pos)
        w = 1.0 / (unc_eps + torch.clamp(unc, min=0.0))
        if unc_power != 1.0:
            w = w.pow(unc_power)

        w_mean = w.mean(dim=(-2, -1), keepdim=True)
        w = w / (w_mean + 1e-6)
        if unc_max_weight > 0:
            w = torch.clamp(w, max=float(unc_max_weight))
        hetero = (w * (output - target_center) ** 2).mean()
        if noisy_input is None:
            return hetero
        input_center = _select_center_if_burst(noisy_input, burst_size, target_pos)

        w_input = 1.0 / (w + 1e-6)
        w_input_mean = w_input.mean(dim=(-2, -1), keepdim=True)
        w_input = w_input / (w_input_mean + 1e-6)
        if unc_max_weight > 0:
            w_input = torch.clamp(w_input, max=float(unc_max_weight))
        input_consistency = (w_input * (output - input_center) ** 2).mean()
        return hetero + float(OPT_RISK_INPUT_WEIGHT) * input_consistency
    mask = mask.bool()
    expanded = mask.expand_as(output)
    if bool(expanded.any()):
        return nn.functional.mse_loss(output[expanded], target_center[expanded])
    return nn.functional.mse_loss(output, target_center)


def _compute_perceptual_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    burst_size: int,
    perceptual_loss_fn: nn.Module,
    target_pos: int | None = None,
) -> torch.Tensor:
    output_center = _select_center_if_burst(output, burst_size, target_pos)
    target_center = _select_center_if_burst(target, burst_size, target_pos)
    return perceptual_loss_fn(output_center, target_center)


def _infer_dataset_tag(root: Path, dataset_kind: str = "auto") -> str:
    return "jhu"


def _make_noise_cfg(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_LOW_DOSE_CFG)
    for key, default in cfg.items():
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
    if cfg.get("peak") is None or cfg.get("sigma_read") is None:
        raise ValueError(
            "DEFAULT_LOW_DOSE_CFG['peak'] and ['sigma_read'] are not set. "
            "Run `python peak_sigma_estimate.py` in `paper_code/`, then copy "
            "`estimate_peak` and `estimate_sigma_read` into this dataset's data.py."
        )
    return dict(DEFAULT_LOW_DOSE_CFG)


def _run_model(
    model: torch.nn.Module, inp: torch.Tensor, noise_cfg: dict
) -> torch.Tensor:
    if isinstance(model, (FastDVDnet, FFDNet)):
        noise_level = get_noise_level(noise_cfg)
        h, w = inp.size(-2), inp.size(-1)
        noise_map = torch.full(
            (inp.size(0), 1, h, w),
            noise_level,
            device=inp.device,
            dtype=inp.dtype,
        )
        return model(inp, noise_map)
    return model(inp)


def _load_pretrained(
    model: torch.nn.Module, args: argparse.Namespace, dataset_tag: str
) -> None:
    if not args.use_pretrained and not args.pretrained_path:
        return
    if args.pretrained_path:
        ckpt_path = Path(args.pretrained_path)
    else:
        dose_tag = _fmt_float(args.dose_fraction)
        peak_tag = _fmt_float(args.peak)
        ckpt_path = (
            Path("saved_model")
            / f"pre_{dataset_tag}"
            / f"{args.model}-pretrain-{dose_tag}-{peak_tag}.pt"
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    src_state = ckpt.get("model", {})
    dst_state = model.state_dict()

    filtered_state = {}
    skipped_shape = 0
    for key, value in src_state.items():
        if key not in dst_state:
            continue
        if dst_state[key].shape != value.shape:
            skipped_shape += 1
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading pretrained: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading pretrained: {len(unexpected)}")
    if skipped_shape:
        print(f"[warn] Skipped pretrained keys due to shape mismatch: {skipped_shape}")


def _identity_weight(epoch: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return float(start)
    t = float(epoch - 1) / float(total - 1)
    return float(start) + (float(end) - float(start)) * t


def _fdk_temporal_y0(model: torch.nn.Module, inp: torch.Tensor) -> torch.Tensor | None:
    if isinstance(model, FDK):
        if inp.dim() == 5:
            x = inp[:, :, :1, ...]
            target = x[:, -1]
            if model.temporal_input_fusion and model._temporal_enabled:
                mu = model._temporal_fuse_mu(x, target)
            else:
                mu = target
            v = model._reliability_map(x, target)
            y0 = model._kalman_update(mu, target, v)
            return model.out_proj(y0)
        target = inp[:, :1, ...]
        mu = target
        v = torch.zeros_like(target)
        y0 = model._kalman_update(mu, target, v)
        return model.out_proj(y0)
    if hasattr(model, "forward_temporal"):
        return model.forward_temporal(inp)
    return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_one_epoch(
    model,
    loader,
    device,
    mode,
    noise_cfg,
    burst_size,
    opt,
    *,
    use_risk: bool = False,
    unc_eps: float = 1e-3,
    unc_power: float = 1.0,
    unc_max_weight: float = 4.0,
    perceptual_loss_fn: nn.Module | None = None,
    perceptual_weight: float = 0.0,
    opt_l1_weight: float = 0.0,
    opt_ssim_weight: float = 0.0,
    target_pos: int | None = None,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        inp, target, mask, unc = unpack_batch(batch, mode, use_risk)
        inp = inp.to(device)
        target = target.to(device)
        mask = mask.to(device) if mask is not None else None
        unc = unc.to(device) if unc is not None else None
        output = _run_model(model, inp, noise_cfg)
        loss = compute_loss(
            output,
            target,
            mask,
            burst_size=burst_size,
            target_pos=target_pos,
            risk=unc if use_risk else None,
            noisy_input=inp,
            unc_eps=unc_eps,
            unc_power=unc_power,
            unc_max_weight=unc_max_weight,
        )
        if perceptual_loss_fn is not None and perceptual_weight > 0.0:
            loss = loss + perceptual_weight * _compute_perceptual_loss(
                output,
                target,
                burst_size,
                perceptual_loss_fn,
                target_pos=target_pos,
            )
        if mode == "opt+r":
            out_center = _select_center_if_burst(output, burst_size, target_pos)
            tgt_center = _select_center_if_burst(target, burst_size, target_pos)
            if opt_l1_weight > 0:
                loss = loss + float(opt_l1_weight) * nn.functional.l1_loss(
                    out_center, tgt_center
                )
            if opt_ssim_weight > 0:
                loss = loss + float(opt_ssim_weight) * _ssim_loss(
                    torch.clamp(out_center, 0.0, 1.0),
                    torch.clamp(tgt_center, 0.0, 1.0),
                )
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        count += 1
    return total / max(1, count)


def _temporal_smoothness_loss(
    model: torch.nn.Module,
    inp: torch.Tensor,
    output: torch.Tensor,
    target_pos: int | None = None,
) -> torch.Tensor:
    if not hasattr(model, "forward_spatial"):
        return output.new_tensor(0.0)
    if inp.dim() != 5 or output.dim() != 5:
        return output.new_tensor(0.0)
    if inp.size(1) < 3:
        return output.new_tensor(0.0)

    pos = (
        inp.size(1) // 2
        if target_pos is None
        else max(0, min(int(target_pos), inp.size(1) - 1))
    )
    center = inp[:, pos]
    center_spatial = model.forward_spatial(center)
    center_temporal = output[:, pos]
    return nn.functional.mse_loss(center_temporal, center_spatial)


def _eval_one_epoch(
    model,
    loader,
    device,
    mode,
    noise_cfg,
    burst_size,
    *,
    use_risk: bool = False,
    unc_eps: float = 1e-3,
    unc_power: float = 1.0,
    unc_max_weight: float = 4.0,
    perceptual_loss_fn: nn.Module | None = None,
    perceptual_weight: float = 0.0,
    opt_l1_weight: float = 0.0,
    opt_ssim_weight: float = 0.0,
    target_pos: int | None = None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            inp, target, mask, unc = unpack_batch(batch, mode, use_risk)
            inp = inp.to(device)
            target = target.to(device)
            mask = mask.to(device) if mask is not None else None
            unc = unc.to(device) if unc is not None else None
            output = _run_model(model, inp, noise_cfg)
            loss = compute_loss(
                output,
                target,
                mask,
                burst_size=burst_size,
                target_pos=target_pos,
                risk=unc if use_risk else None,
                noisy_input=inp,
                unc_eps=unc_eps,
                unc_power=unc_power,
                unc_max_weight=unc_max_weight,
            )
            if perceptual_loss_fn is not None and perceptual_weight > 0.0:
                loss = loss + perceptual_weight * _compute_perceptual_loss(
                    output,
                    target,
                    burst_size,
                    perceptual_loss_fn,
                    target_pos=target_pos,
                )
            if mode == "opt+r":
                out_center = _select_center_if_burst(output, burst_size, target_pos)
                tgt_center = _select_center_if_burst(target, burst_size, target_pos)
                if opt_l1_weight > 0:
                    loss = loss + float(opt_l1_weight) * nn.functional.l1_loss(
                        out_center, tgt_center
                    )
                if opt_ssim_weight > 0:
                    loss = loss + float(opt_ssim_weight) * _ssim_loss(
                        torch.clamp(out_center, 0.0, 1.0),
                        torch.clamp(tgt_center, 0.0, 1.0),
                    )
            total += loss.item()
            count += 1
    return total / max(1, count)


def _gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    b = real.size(0)
    alpha = torch.rand((b, 1, 1, 1), device=device, dtype=real.dtype)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interp)
    ones = torch.ones_like(d_interp, device=device, dtype=real.dtype)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(b, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def _train_one_epoch_wgan(
    generator: nn.Module,
    discriminator: nn.Module,
    loader,
    device: torch.device,
    noise_cfg: dict,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    *,
    adv_weight: float,
    gp_weight: float,
    d_steps: int,
    l1_weight: float,
    perceptual_loss_fn: nn.Module | None,
    perceptual_weight: float,
) -> tuple[float, float]:
    generator.train()
    discriminator.train()
    g_total = 0.0
    d_total = 0.0
    count = 0

    for inp, target in loader:
        inp = inp.to(device)
        target = target.to(device)

        for _ in range(max(1, int(d_steps))):
            with torch.no_grad():
                fake_detached = _run_model(generator, inp, noise_cfg).detach()
            d_real = discriminator(target).mean()
            d_fake = discriminator(fake_detached).mean()
            gp = _gradient_penalty(discriminator, target, fake_detached, device)
            d_loss = d_fake - d_real + float(gp_weight) * gp
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            d_total += float(d_loss.item())

        fake = _run_model(generator, inp, noise_cfg)
        adv = -discriminator(fake).mean()
        rec = nn.functional.l1_loss(fake, target)
        g_loss = float(l1_weight) * rec + float(adv_weight) * adv
        if perceptual_loss_fn is not None and perceptual_weight > 0.0:
            g_loss = g_loss + float(perceptual_weight) * perceptual_loss_fn(
                fake, target
            )
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        g_total += float(g_loss.item())
        count += 1

    denom = max(1, count)
    return g_total / denom, d_total / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset_kind", default="jhu", choices=["jhu"])
    parser.add_argument(
        "--model",
        default="unet",
        choices=[
            "unet",
            "unet_tmp",
            "fastdvdnet",
            "bm3d",
            "nlm",
            "fdk",
            "dncnn",
            "redcnn",
            "red-cnn",
            "wganvgg",
            "wgan-vgg",
            "ffdnet",
            "nafnet",
            "edvr",
        ],
    )
    parser.add_argument(
        "--mode",
        default="n2c",
        choices=["n2c", "n2v", "n2self", "r2r", "pretrain", "opt+r"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.2)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--video_burst", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--unet_base_channels", type=int, default=16)
    parser.add_argument("--unet_depth", type=int, default=3)
    parser.add_argument("--dncnn_depth", type=int, default=17)
    parser.add_argument("--dncnn_features", type=int, default=64)
    parser.add_argument("--fastdvdnet_base_channels", type=int, default=32)
    parser.add_argument("--redcnn_channels", type=int, default=96)
    parser.add_argument("--redcnn_layers", type=int, default=5)
    parser.add_argument("--wgan_base_channels", type=int, default=32)
    parser.add_argument("--wgan_blocks", type=int, default=8)
    parser.add_argument("--wgan_d_lr", type=float, default=None)
    parser.add_argument("--wgan_adv_weight", type=float, default=1.0)
    parser.add_argument("--wgan_gp_weight", type=float, default=10.0)
    parser.add_argument("--wgan_d_steps", type=int, default=4)
    parser.add_argument("--wgan_l1_weight", type=float, default=0.0)
    parser.add_argument("--wgan_perceptual_weight", type=float, default=0.1)
    parser.add_argument("--ffdnet_features", type=int, default=64)
    parser.add_argument("--ffdnet_depth", type=int, default=15)
    parser.add_argument("--nafnet_width", type=int, default=32)
    parser.add_argument("--nafnet_blocks", type=int, default=6)
    parser.add_argument("--edvr_channels", type=int, default=32)
    parser.add_argument("--edvr_blocks", type=int, default=10)
    parser.add_argument("--fdk_width", type=int, default=32)
    parser.add_argument("--fdk_blocks", type=int, default=4)
    parser.add_argument("--fdk_radius", type=int, default=1)
    parser.add_argument("--fdk_eps", type=float, default=1e-4)
    parser.add_argument("--fdk_use_var", action="store_true")
    parser.add_argument("--fdk_disable_kalman", action="store_true")
    parser.add_argument(
        "--fdk_stage1_mode_opt",
        default="opt+r",
        choices=["opt+r", "n2v", "n2self", "r2r"],
    )
    parser.add_argument("--fdk_stage2_joint_spatial", action="store_true")
    parser.add_argument("--fdk_stage2_lr_scale", type=float, default=0.3)
    parser.add_argument("--fdk_temporal_progressive", action="store_true")
    parser.add_argument("--fdk_temporal_level_epochs", type=int, default=None)
    parser.add_argument("--fdk_temporal_joint_epochs", type=int, default=0)
    parser.add_argument("--fdk_identity_start", type=float, default=0.0)
    parser.add_argument("--fdk_identity_end", type=float, default=0.0)
    parser.add_argument("--fdk_temporal_smoothness_weight", type=float, default=0.0)

    parser.add_argument(
        "--dose_fraction", type=float, default=DEFAULT_LOW_DOSE_CFG["dose_fraction"]
    )
    parser.add_argument("--peak", type=float, default=DEFAULT_LOW_DOSE_CFG["peak"])
    parser.add_argument(
        "--sigma_read", type=float, default=DEFAULT_LOW_DOSE_CFG["sigma_read"]
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_LOW_DOSE_CFG["gamma"])
    parser.add_argument(
        "--corr_sigma", type=float, default=DEFAULT_LOW_DOSE_CFG["corr_sigma"]
    )
    parser.add_argument(
        "--corr_alpha", type=float, default=DEFAULT_LOW_DOSE_CFG["corr_alpha"]
    )
    parser.add_argument(
        "--corr_mix", type=float, default=DEFAULT_LOW_DOSE_CFG["corr_mix"]
    )
    parser.add_argument(
        "--stripe_amp", type=float, default=DEFAULT_LOW_DOSE_CFG["stripe_amp"]
    )
    parser.add_argument(
        "--stripe_axis",
        default=DEFAULT_LOW_DOSE_CFG["stripe_axis"],
        choices=["col", "row"],
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=DEFAULT_LOW_DOSE_CFG["mask_ratio"]
    )
    parser.add_argument(
        "--noise_level", type=float, default=DEFAULT_LOW_DOSE_CFG["noise_level"]
    )

    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("-use_pretrained", dest="use_pretrained", action="store_true")
    parser.add_argument("--pretrained_path", default=None)

    parser.add_argument("--opt_epochs", type=int, default=None)
    parser.add_argument("--opt_lr", type=float, default=None)
    parser.add_argument("--opt_burst", type=int, default=5)
    parser.add_argument("--opt_label_root", default="auto")
    parser.add_argument("--opt_use_risk", action="store_true")
    parser.add_argument("--opt_risk_eps", type=float, default=1e-3)
    parser.add_argument("--opt_risk_power", type=float, default=1.0)
    parser.add_argument("--opt_risk_max_weight", type=float, default=4.0)
    parser.add_argument("--opt_risk_input_weight", type=float, default=0.1)
    parser.add_argument("--opt_label_jitter", type=float, default=0.0)
    parser.add_argument("--opt_l1_weight", type=float, default=0.0)
    parser.add_argument("--opt_ssim_weight", type=float, default=0.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.0)
    parser.set_defaults(
        fdk_stage2_joint_spatial=True,
        fdk_temporal_progressive=True,
    )
    args = parser.parse_args()
    args.mode = _normalize_mode_alias(args.mode)
    args.fdk_stage1_mode_opt = _normalize_mode_alias(args.fdk_stage1_mode_opt)
    if _is_opt_mode(args.mode):
        args.opt_use_risk = True

    global OPT_RISK_INPUT_WEIGHT
    OPT_RISK_INPUT_WEIGHT = float(args.opt_risk_input_weight)

    if args.epochs is None:
        args.epochs = 30 if args.model.lower() in {"fdk", "unet_tmp"} else 50
    if args.model.lower() in {"fdk", "unet_tmp"} and args.opt_epochs is None:
        args.opt_epochs = 30

    if args.seed is not None:
        _set_seed(int(args.seed))

    if args.dataset_kind == "auto":
        args.dataset_kind = _infer_dataset_tag(Path(args.root), "auto")
    if args.opt_label_root in {None, "", "auto"}:
        args.opt_label_root = f"opt_label_{args.dataset_kind}"

    noise_cfg = _make_noise_cfg(args)
    device = torch.device(args.device)
    dataset_tag = _infer_dataset_tag(Path(args.root), args.dataset_kind)

    fdk_kwargs = dict(
        width=args.fdk_width,
        blocks=args.fdk_blocks,
        radius=args.fdk_radius,
        down=1,
        eps=args.fdk_eps,
        temporal_input_fusion=True,
        temporal_center_only=True,
        temporal_attn_temp=0.1,
        use_var=args.fdk_use_var,
        disable_kalman=args.fdk_disable_kalman,
    )
    model_presets = {
        "unet": {"base_channels": args.unet_base_channels, "depth": args.unet_depth},
        "unet_tmp": {
            "base_channels": args.unet_base_channels,
            "depth": args.unet_depth,
            "temporal_attn_temp": 0.1,
            "temporal_input_fusion": True,
            "temporal_center_only": True,
        },
        "fastdvdnet": {
            "num_frames": args.video_burst,
            "default_noise_level": args.noise_level,
            "base_channels": args.fastdvdnet_base_channels,
        },
        "dncnn": {"depth": args.dncnn_depth, "features": args.dncnn_features},
        "redcnn": {"channels": args.redcnn_channels, "num_layers": args.redcnn_layers},
        "red-cnn": {"channels": args.redcnn_channels, "num_layers": args.redcnn_layers},
        "wganvgg": {
            "base_channels": args.wgan_base_channels,
            "num_blocks": args.wgan_blocks,
        },
        "wgan-vgg": {
            "base_channels": args.wgan_base_channels,
            "num_blocks": args.wgan_blocks,
        },
        "ffdnet": {
            "features": args.ffdnet_features,
            "depth": args.ffdnet_depth,
            "default_noise_level": args.noise_level,
        },
        "nafnet": {"width": args.nafnet_width, "blocks": args.nafnet_blocks},
        "edvr": {
            "num_frames": args.video_burst,
            "channels": args.edvr_channels,
            "num_blocks": args.edvr_blocks,
        },
        "fdk": fdk_kwargs,
    }
    model_name = args.model.lower()
    model_kwargs = dict(model_presets.get(model_name, {}))
    model = build_model(
        args.model,
        in_channels=args.in_channels,
        **model_kwargs,
    ).to(device)

    if args.mode != "pretrain":
        _load_pretrained(model, args, dataset_tag)

    perceptual_loss_fn = None
    if args.perceptual_weight > 0.0 and args.mode in {"pretrain", "opt+r"}:
        perceptual_loss_fn = VGGPerceptualLoss().to(device)
        perceptual_loss_fn.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Model has no trainable parameters; skipping training.")
        return

    extra_ckpt: dict = {}

    if isinstance(model, (FDK, UNetTemp)):
        stage_prefix = "fdk" if isinstance(model, FDK) else "unet_tmp"
        stage1_mode_opt = args.fdk_stage1_mode_opt
        stage2_joint_spatial = args.fdk_stage2_joint_spatial
        stage2_lr_scale = max(1e-6, float(args.fdk_stage2_lr_scale))
        temporal_smoothness_weight = args.fdk_temporal_smoothness_weight
        joint_epochs = max(0, int(args.fdk_temporal_joint_epochs))

        stage1_mode = stage1_mode_opt if args.mode == "opt+r" else args.mode
        stage1_pretrain = stage1_mode == "pretrain"
        stage1_use_risk = (
            args.opt_use_risk if stage1_mode == "opt+r" else False
        )
        stage1_use_perceptual = _use_perceptual(stage1_mode, args.perceptual_weight)
        model.set_temporal_trainable(False)
        model.set_spatial_trainable(True)
        model.set_temporal_enabled(False)

        train_loader, val_loader, _ = get_dataloaders(
            root=args.root,
            dataset_kind=args.dataset_kind,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode=stage1_mode,
            in_channels=args.in_channels,
            noise_cfg=noise_cfg,
            burst_size=1,
            pretrain=stage1_pretrain,
            opt_label_root=args.opt_label_root if stage1_mode == "opt+r" else None,
            use_opt_risk=stage1_use_risk,
            opt_label_jitter=args.opt_label_jitter if stage1_mode == "opt+r" else 0.0,
        )

        opt = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        scheduler = StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_one_epoch(
                model,
                train_loader,
                device,
                stage1_mode,
                noise_cfg,
                1,
                opt,
                use_risk=stage1_use_risk,
                unc_eps=args.opt_risk_eps,
                unc_power=args.opt_risk_power,
                unc_max_weight=args.opt_risk_max_weight,
                perceptual_loss_fn=perceptual_loss_fn
                if stage1_use_perceptual
                else None,
                perceptual_weight=args.perceptual_weight
                if stage1_use_perceptual
                else 0.0,
                opt_l1_weight=args.opt_l1_weight if stage1_mode == "opt+r" else 0.0,
                opt_ssim_weight=args.opt_ssim_weight if stage1_mode == "opt+r" else 0.0,
            )
            val_loss = _eval_one_epoch(
                model,
                val_loader,
                device,
                stage1_mode,
                noise_cfg,
                1,
                use_risk=stage1_use_risk,
                unc_eps=args.opt_risk_eps,
                unc_power=args.opt_risk_power,
                unc_max_weight=args.opt_risk_max_weight,
                perceptual_loss_fn=perceptual_loss_fn
                if stage1_use_perceptual
                else None,
                perceptual_weight=args.perceptual_weight
                if stage1_use_perceptual
                else 0.0,
                opt_l1_weight=args.opt_l1_weight if stage1_mode == "opt+r" else 0.0,
                opt_ssim_weight=args.opt_ssim_weight if stage1_mode == "opt+r" else 0.0,
            )
            scheduler.step()
            print(
                f"[{stage_prefix}-stage1] Epoch {epoch}/{args.epochs} - train: {train_loss:.6f} val: {val_loss:.6f}"
            )

        model.set_spatial_trainable(stage2_joint_spatial)
        model.set_temporal_trainable(True)
        model.set_temporal_enabled(True)

        stage2_mode = "opt+r" if args.mode == "opt+r" else args.mode
        stage2_pretrain = stage2_mode == "pretrain"
        stage2_use_risk = (
            args.opt_use_risk if stage2_mode == "opt+r" else False
        )
        stage2_use_perceptual = _use_perceptual(stage2_mode, args.perceptual_weight)
        stage2_epochs = args.opt_epochs or args.epochs
        stage2_lr = (args.opt_lr or args.lr) * stage2_lr_scale
        stage2_burst = args.opt_burst
        stage2_target_pos = max(0, int(stage2_burst) - 1)

        train_loader, val_loader, _ = get_dataloaders(
            root=args.root,
            dataset_kind=args.dataset_kind,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode=stage2_mode,
            in_channels=args.in_channels,
            noise_cfg=noise_cfg,
            burst_size=stage2_burst,
            pretrain=stage2_pretrain,
            opt_label_root=args.opt_label_root if stage2_mode == "opt+r" else None,
            use_opt_risk=stage2_use_risk,
            opt_label_jitter=args.opt_label_jitter if stage2_mode == "opt+r" else 0.0,
            burst_align=False,
            burst_align_max_shift=10.0,
            burst_causal=True,
            burst_target="last",
        )

        def _run_temporal_epochs(
            epochs: int, epoch_offset: int, total_epochs: int, label: str
        ) -> int:
            opt2 = Adam(
                [p for p in model.parameters() if p.requires_grad], lr=stage2_lr
            )
            scheduler2 = StepLR(opt2, step_size=args.lr_step_size, gamma=args.lr_gamma)
            for epoch in range(1, epochs + 1):
                model.train()
                total = 0.0
                count = 0
                global_epoch = epoch_offset + epoch
                for batch in train_loader:
                    inp, target, mask, unc = unpack_batch(
                        batch, stage2_mode, stage2_use_risk
                    )
                    inp = inp.to(device)
                    target = target.to(device)
                    mask = mask.to(device) if mask is not None else None
                    unc = unc.to(device) if unc is not None else None
                    output = _run_model(model, inp, noise_cfg)
                    loss = compute_loss(
                        output,
                        target,
                        mask,
                        burst_size=stage2_burst,
                        target_pos=stage2_target_pos,
                        risk=unc if stage2_use_risk else None,
                        noisy_input=inp,
                        unc_eps=args.opt_risk_eps,
                        unc_power=args.opt_risk_power,
                        unc_max_weight=args.opt_risk_max_weight,
                    )
                    y0 = _fdk_temporal_y0(model, inp)
                    if y0 is not None:
                        loss = loss + compute_loss(
                            y0,
                            target,
                            mask,
                            burst_size=stage2_burst,
                            target_pos=stage2_target_pos,
                            risk=unc if stage2_use_risk else None,
                            noisy_input=inp,
                            unc_eps=args.opt_risk_eps,
                            unc_power=args.opt_risk_power,
                            unc_max_weight=args.opt_risk_max_weight,
                        )
                    if stage2_use_perceptual:
                        loss = loss + args.perceptual_weight * _compute_perceptual_loss(
                            output,
                            target,
                            stage2_burst,
                            perceptual_loss_fn,
                            target_pos=stage2_target_pos,
                        )
                    if temporal_smoothness_weight > 0:
                        loss = (
                            loss
                            + temporal_smoothness_weight
                            * _temporal_smoothness_loss(
                                model,
                                inp,
                                output,
                                target_pos=stage2_target_pos,
                            )
                        )
                    opt2.zero_grad()
                    loss.backward()
                    opt2.step()
                    total += loss.item()
                    count += 1
                train_loss = total / max(1, count)
                val_loss = _eval_one_epoch(
                    model,
                    val_loader,
                    device,
                    stage2_mode,
                    noise_cfg,
                    stage2_burst,
                    use_risk=stage2_use_risk,
                    target_pos=stage2_target_pos,
                    unc_eps=args.opt_risk_eps,
                    unc_power=args.opt_risk_power,
                    unc_max_weight=args.opt_risk_max_weight,
                    perceptual_loss_fn=perceptual_loss_fn
                    if stage2_use_perceptual
                    else None,
                    perceptual_weight=args.perceptual_weight
                    if stage2_use_perceptual
                    else 0.0,
                    opt_l1_weight=args.opt_l1_weight if stage2_mode == "opt+r" else 0.0,
                    opt_ssim_weight=args.opt_ssim_weight
                    if stage2_mode == "opt+r"
                    else 0.0,
                )
                scheduler2.step()
                print(
                    f"[{stage_prefix}-stage2:{label}] Epoch {global_epoch}/{total_epochs} "
                    f"- train: {train_loss:.6f} val: {val_loss:.6f}"
                )
            return epoch_offset + epochs

        total_epochs = stage2_epochs + joint_epochs
        epoch_offset = 0

        temporal_keys = model.temporal_keys(deep_to_shallow=True)
        if args.fdk_temporal_progressive and len(temporal_keys) > 1:
            if args.fdk_temporal_level_epochs is not None:
                per_epochs = max(0, int(args.fdk_temporal_level_epochs))
                epochs_list = [per_epochs for _ in temporal_keys]
                stage2_epochs = per_epochs * len(temporal_keys)
                total_epochs = stage2_epochs + joint_epochs
            else:
                per_epochs = stage2_epochs // len(temporal_keys)
                remainder = stage2_epochs % len(temporal_keys)
                epochs_list = [
                    per_epochs + (1 if i < remainder else 0)
                    for i in range(len(temporal_keys))
                ]
            for key, epochs_i in zip(temporal_keys, epochs_list):
                if epochs_i <= 0:
                    continue
                model.set_temporal_trainable_keys([key])
                epoch_offset = _run_temporal_epochs(
                    epochs_i, epoch_offset, total_epochs, key
                )
            if args.fdk_temporal_joint_epochs > 0:
                model.set_temporal_trainable(True)
                epoch_offset = _run_temporal_epochs(
                    int(args.fdk_temporal_joint_epochs),
                    epoch_offset,
                    total_epochs,
                    "joint",
                )
        else:
            if joint_epochs > 0:
                stage2_epochs = stage2_epochs + joint_epochs
                total_epochs = stage2_epochs
            model.set_temporal_trainable(True)
            _run_temporal_epochs(stage2_epochs, 0, total_epochs, "all")

    elif model_name in {"wganvgg", "wgan-vgg"} and args.mode == "n2c":
        burst_size = 1
        train_loader, val_loader, _ = get_dataloaders(
            root=args.root,
            dataset_kind=args.dataset_kind,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode=args.mode,
            in_channels=args.in_channels,
            noise_cfg=noise_cfg,
            burst_size=burst_size,
            pretrain=False,
            opt_label_root=None,
            use_opt_risk=False,
            opt_label_jitter=0.0,
        )
        discriminator = WGANVGGDiscriminator(
            input_size=int(args.patch_size), in_channels=args.in_channels
        ).to(device)
        wgan_perc = None
        if args.wgan_perceptual_weight > 0:
            wgan_perc = VGGPerceptualLoss().to(device)
            wgan_perc.eval()

        g_opt = Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
        d_opt = Adam(
            discriminator.parameters(), lr=args.wgan_d_lr or args.lr, betas=(0.5, 0.9)
        )
        g_scheduler = StepLR(g_opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
        d_scheduler = StepLR(d_opt, step_size=args.lr_step_size, gamma=args.lr_gamma)

        for epoch in range(1, args.epochs + 1):
            g_loss, d_loss = _train_one_epoch_wgan(
                model,
                discriminator,
                train_loader,
                device,
                noise_cfg,
                g_opt,
                d_opt,
                adv_weight=args.wgan_adv_weight,
                gp_weight=args.wgan_gp_weight,
                d_steps=args.wgan_d_steps,
                l1_weight=args.wgan_l1_weight,
                perceptual_loss_fn=wgan_perc,
                perceptual_weight=args.wgan_perceptual_weight,
            )
            val_loss = _eval_one_epoch(
                model,
                val_loader,
                device,
                args.mode,
                noise_cfg,
                burst_size,
                use_risk=False,
                unc_eps=args.opt_risk_eps,
                unc_power=args.opt_risk_power,
                unc_max_weight=args.opt_risk_max_weight,
                perceptual_loss_fn=None,
                perceptual_weight=0.0,
                opt_l1_weight=0.0,
                opt_ssim_weight=0.0,
            )
            g_scheduler.step()
            d_scheduler.step()
            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"G: {g_loss:.6f} D: {d_loss:.6f} val(mse): {val_loss:.6f}"
            )
        extra_ckpt["discriminator"] = discriminator.state_dict()

    else:
        burst_size = (
            args.video_burst if args.model.lower() in {"fastdvdnet", "edvr"} else 1
        )
        pretrain = args.mode == "pretrain"
        use_risk = args.opt_use_risk if args.mode == "opt+r" else False
        use_perceptual = _use_perceptual(args.mode, args.perceptual_weight)

        train_loader, val_loader, _ = get_dataloaders(
            root=args.root,
            dataset_kind=args.dataset_kind,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode=args.mode,
            in_channels=args.in_channels,
            noise_cfg=noise_cfg,
            burst_size=burst_size,
            pretrain=pretrain,
            opt_label_root=args.opt_label_root if args.mode == "opt+r" else None,
            use_opt_risk=use_risk,
            opt_label_jitter=args.opt_label_jitter if args.mode == "opt+r" else 0.0,
        )

        opt = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        scheduler = StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)

        for epoch in range(1, args.epochs + 1):
            train_loss = _train_one_epoch(
                model,
                train_loader,
                device,
                args.mode,
                noise_cfg,
                burst_size,
                opt,
                use_risk=use_risk,
                unc_eps=args.opt_risk_eps,
                unc_power=args.opt_risk_power,
                unc_max_weight=args.opt_risk_max_weight,
                perceptual_loss_fn=perceptual_loss_fn if use_perceptual else None,
                perceptual_weight=args.perceptual_weight if use_perceptual else 0.0,
                opt_l1_weight=args.opt_l1_weight if args.mode == "opt+r" else 0.0,
                opt_ssim_weight=args.opt_ssim_weight if args.mode == "opt+r" else 0.0,
            )
            val_loss = _eval_one_epoch(
                model,
                val_loader,
                device,
                args.mode,
                noise_cfg,
                burst_size,
                use_risk=use_risk,
                unc_eps=args.opt_risk_eps,
                unc_power=args.opt_risk_power,
                unc_max_weight=args.opt_risk_max_weight,
                perceptual_loss_fn=perceptual_loss_fn if use_perceptual else None,
                perceptual_weight=args.perceptual_weight if use_perceptual else 0.0,
                opt_l1_weight=args.opt_l1_weight if args.mode == "opt+r" else 0.0,
                opt_ssim_weight=args.opt_ssim_weight if args.mode == "opt+r" else 0.0,
            )
            scheduler.step()
            print(
                f"Epoch {epoch}/{args.epochs} - train: {train_loss:.6f} val: {val_loss:.6f}"
            )

    dataset_tag = _infer_dataset_tag(Path(args.root), args.dataset_kind)
    save_prefix = f"pre_{dataset_tag}" if args.mode == "pretrain" else dataset_tag
    save_dir = Path("saved_model") / save_prefix
    save_dir.mkdir(parents=True, exist_ok=True)

    dose_tag = _fmt_float(args.dose_fraction)
    peak_tag = _fmt_float(args.peak)
    mode_tag = _mode_tag(args.mode, args.opt_use_risk)
    if args.mode != "pretrain" and args.use_pretrained:
        model_tag = f"{args.model}-pretrain-{mode_tag}-{dose_tag}-{peak_tag}"
    else:
        model_tag = f"{args.model}-{mode_tag}-{dose_tag}-{peak_tag}"

    save_path = save_dir / f"{model_tag}.pt"
    if args.model.lower() in {"fdk", "unet_tmp"}:
        burst_size = int(args.opt_burst)
    else:
        burst_size = (
            args.video_burst if args.model.lower() in {"fastdvdnet", "edvr"} else 1
        )
    save_args = dict(vars(args))
    save_args["mode"] = _mode_tag(args.mode, args.opt_use_risk)
    save_args["fdk_down"] = 1
    if args.model.lower() == "unet_tmp":
        save_args["unet_tmp_temporal_attn_temp"] = 0.1
        save_args["unet_tmp_temporal_input_fusion"] = True
        save_args["unet_tmp_temporal_center_only"] = True
    torch.save(
        {
            "model": model.state_dict(),
            "args": save_args,
            "noise_cfg": noise_cfg,
            "dataset": dataset_tag,
            "burst_size": burst_size,
            **extra_ckpt,
        },
        save_path,
    )

    config_path = save_dir / f"{model_tag}.json"
    config_path.write_text(
        json.dumps(
            {
                "args": save_args,
                "noise_cfg": noise_cfg,
                "dataset": dataset_tag,
                "burst_size": burst_size,
            },
            indent=2,
        )
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
