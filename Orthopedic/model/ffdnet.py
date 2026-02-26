import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pixel_unshuffle(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // scale, scale, w // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * scale * scale, h // scale, w // scale)


class FFDNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        features: int = 64,
        depth: int = 15,
        default_noise_level: float = 0.03,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.default_noise_level = float(default_noise_level)
        self.scale = 2

        in_ch = self.in_channels * self.scale * self.scale + 1
        out_ch = self.in_channels * self.scale * self.scale
        layers: list[nn.Module] = []
        layers.append(
            nn.Conv2d(in_ch, features, kernel_size=3, stride=1, padding=1, bias=True)
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(max(1, int(depth) - 2)):
            layers.append(
                nn.Conv2d(
                    features, features, kernel_size=3, stride=1, padding=1, bias=True
                )
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(features, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.net = nn.Sequential(*layers)
        self.up = nn.PixelShuffle(self.scale)

    def _sigma_to_map(
        self, x_down: torch.Tensor, sigma: torch.Tensor | None
    ) -> torch.Tensor:
        b, _c, h, w = x_down.shape
        if sigma is None:
            return torch.full(
                (b, 1, h, w),
                float(self.default_noise_level),
                dtype=x_down.dtype,
                device=x_down.device,
            )
        if (
            sigma.dim() == 4
            and sigma.size(1) == 1
            and sigma.size(-2) == 1
            and sigma.size(-1) == 1
        ):
            return sigma.to(dtype=x_down.dtype, device=x_down.device).repeat(1, 1, h, w)
        if sigma.dim() == 4 and sigma.size(1) == 1:
            return F.interpolate(
                sigma.to(dtype=x_down.dtype, device=x_down.device),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        raise ValueError("sigma/noise_map must be [B,1,1,1] or [B,1,H,W]")

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, x.size(1) // 2]
        if x.dim() != 4:
            raise ValueError(f"FFDNet expects BCHW/BTCHW, got {tuple(x.shape)}")

        h, w = x.shape[-2:]
        pad_h = int(math.ceil(h / 2.0) * 2 - h)
        pad_w = int(math.ceil(w / 2.0) * 2 - w)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x_down = _pixel_unshuffle(x, self.scale)
        sigma_map = self._sigma_to_map(x_down, sigma)
        y = self.net(torch.cat([x_down, sigma_map], dim=1))
        y = self.up(y)
        y = y[..., :h, :w]
        return torch.clamp(y, 0.0, 1.0)
