import torch
import torch.nn as nn
import torch.nn.functional as F


class FDK(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        *,
        width: int = 32,
        blocks: int = 4,
        radius: int = 1,
        down: int = 1,
        eps: float = 1e-4,
        temporal_input_fusion: bool = True,
        temporal_center_only: bool = True,
        temporal_attn_temp: float = 0.1,
        use_var: bool = False,
        disable_kalman: bool = False,
        base_channels: int | None = None,
        depth: int | None = None,
        **_kwargs,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.temporal_input_fusion = bool(temporal_input_fusion)
        self.temporal_center_only = bool(temporal_center_only)
        self.temporal_attn_temp = float(max(temporal_attn_temp, 1e-6))

        if base_channels is not None and (width is None or int(width) <= 0):
            width = max(16, int(base_channels) * 2)
        self.width = int(width)
        self.blocks = int(blocks if depth is None else max(blocks, depth))
        self.down = max(1, int(down))
        self.use_var = bool(use_var)
        self.enable_kalman = not bool(disable_kalman)

        self.k_gate = nn.Conv2d(2, 1, kernel_size=1)
        nn.init.zeros_(self.k_gate.weight)
        nn.init.constant_(self.k_gate.bias, -2.0)

        in_c = 3
        layers: list[nn.Module] = [
            nn.Conv2d(in_c, self.width, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, self.blocks - 1)):
            layers += [
                nn.Conv2d(self.width, self.width, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.lr_body = nn.Sequential(*layers)
        self.lr_head = nn.Conv2d(self.width, 1, 3, padding=1)
        nn.init.zeros_(self.lr_head.weight)
        nn.init.zeros_(self.lr_head.bias)

        self.out_proj = (
            nn.Identity()
            if self.out_channels == 1
            else nn.Conv2d(1, self.out_channels, kernel_size=1)
        )

        self._temporal_enabled = True

    def set_temporal_trainable(self, trainable: bool) -> None:
        req = bool(trainable)
        for p in self.k_gate.parameters():
            p.requires_grad = req

    def set_temporal_trainable_keys(self, keys: list[str] | None) -> None:
        req = bool(keys)
        for p in self.k_gate.parameters():
            p.requires_grad = req

    def set_temporal_enabled(self, enabled: bool) -> None:
        self._temporal_enabled = bool(enabled)

    def set_spatial_trainable(self, trainable: bool) -> None:
        req = bool(trainable)
        for p in self.lr_body.parameters():
            p.requires_grad = req
        for p in self.lr_head.parameters():
            p.requires_grad = req
        for p in self.out_proj.parameters():
            p.requires_grad = req

    def temporal_keys(self, *, deep_to_shallow: bool = True) -> list[str]:
        _ = deep_to_shallow
        return ["input_fuse"]

    def _temporal_fuse_mu(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_exp = target.unsqueeze(1)
        dist = torch.mean(torch.abs(x - target_exp), dim=2, keepdim=True)
        logits = -dist / float(self.temporal_attn_temp)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights * x, dim=1)

    def _reliability_map(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_var:
            return torch.var(x, dim=1, unbiased=False)
        target_exp = target.unsqueeze(1)
        return torch.mean(torch.abs(x - target_exp), dim=1)

    def _kalman_update(
        self, mu: torch.Tensor, target: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if not (self.enable_kalman and self._temporal_enabled):
            return mu
        delta = torch.abs(target - mu)
        k = torch.sigmoid(self.k_gate(torch.cat([v, delta], dim=1)))
        return mu + k * (target - mu)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if int(self.down) <= 1:
            return x
        return F.interpolate(x, scale_factor=1.0 / float(self.down), mode="area")

    @staticmethod
    def _upsample(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    def _predict_residual(
        self, y0: torch.Tensor, v: torch.Tensor, target: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        inp_lr = torch.cat([y0, v, target - mu], dim=1)
        inp_lr = self._downsample(inp_lr)
        feat = self.lr_body(inp_lr)
        r_lr = self.lr_head(feat)
        r_hr = self._upsample(r_lr, size_hw=(target.size(-2), target.size(-1)))
        return r_hr

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :1, ...]
        mu = target
        v = torch.zeros_like(target)
        y0 = self._kalman_update(mu, target, v)
        out = y0 + self._predict_residual(y0, v, target, mu)
        return self.out_proj(out)

    def _forward_burst_last(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("FDK expects [B,T,C,H,W] for burst input")
        x = x[:, :, :1, ...]
        target = x[:, -1]
        if self.temporal_input_fusion and self._temporal_enabled:
            mu = self._temporal_fuse_mu(x, target)
        else:
            mu = target
        v = self._reliability_map(x, target)
        y0 = self._kalman_update(mu, target, v)
        out = y0 + self._predict_residual(y0, v, target, mu)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            if self.temporal_center_only:
                return self._forward_burst_last(x)
            outs = []
            for t in range(x.size(1)):
                outs.append(self._forward_single(x[:, t, :1, ...]))
            return torch.stack(outs, dim=1)
        return self._forward_single(x)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            outs = []
            for t in range(x.size(1)):
                outs.append(self._forward_single(x[:, t, :1, ...]))
            return torch.stack(outs, dim=1)
        return self._forward_single(x)
