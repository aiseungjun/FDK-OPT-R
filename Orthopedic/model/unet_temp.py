import torch
import torch.nn as nn

from .unet import UNet


class UNetTemp(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        *,
        base_channels: int = 32,
        depth: int = 3,
        temporal_attn_temp: float = 0.1,
        temporal_input_fusion: bool = True,
        temporal_center_only: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.temporal_input_fusion = bool(temporal_input_fusion)
        self.temporal_center_only = bool(temporal_center_only)
        self.temporal_attn_temp = float(max(temporal_attn_temp, 1e-6))

        gate_hidden = max(8, self.in_channels * 4)
        self.temporal_gate = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=3, padding=1),
        )

        self.spatial = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            base_channels=int(base_channels),
            depth=int(depth),
        )

        self._temporal_enabled = True

    def set_temporal_trainable(self, trainable: bool) -> None:
        req = bool(trainable)
        for p in self.temporal_gate.parameters():
            p.requires_grad = req

    def set_temporal_trainable_keys(self, keys: list[str] | None) -> None:
        self.set_temporal_trainable(bool(keys))

    def set_temporal_enabled(self, enabled: bool) -> None:
        self._temporal_enabled = bool(enabled)

    def set_spatial_trainable(self, trainable: bool) -> None:
        req = bool(trainable)
        for p in self.spatial.parameters():
            p.requires_grad = req

    def temporal_keys(self, *, deep_to_shallow: bool = True) -> list[str]:
        _ = deep_to_shallow
        return ["input_fuse"]

    def _fuse_burst_last(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("UNetTemp expects [B,T,C,H,W] for burst input")
        target = x[:, -1]
        if not (self.temporal_input_fusion and self._temporal_enabled):
            return target

        target_exp = target.unsqueeze(1)
        logits = []
        for t in range(x.size(1)):
            pair = torch.cat([x[:, t], target], dim=1)
            logits.append(self.temporal_gate(pair))
        logit_stack = torch.stack(logits, dim=1)

        dist = torch.mean(torch.abs(x - target_exp), dim=2, keepdim=True)
        logit_stack = logit_stack - dist / self.temporal_attn_temp
        weights = torch.softmax(logit_stack, dim=1)
        fused = torch.sum(weights * x, dim=1)
        return fused

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x)

    def _forward_burst_last(self, x: torch.Tensor) -> torch.Tensor:
        fused = self._fuse_burst_last(x)
        return self.spatial(fused)

    def forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            return self._fuse_burst_last(x)
        if x.dim() == 4:
            return x
        raise ValueError(f"Expected BCHW or BTCHW input, got shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            if self.temporal_center_only:
                return self._forward_burst_last(x)
            outs = []
            for t in range(x.size(1)):
                outs.append(self._forward_single(x[:, t]))
            return torch.stack(outs, dim=1)
        if x.dim() == 4:
            return self._forward_single(x)
        raise ValueError(f"Expected BCHW or BTCHW input, got shape {tuple(x.shape)}")

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            outs = []
            for t in range(x.size(1)):
                outs.append(self._forward_single(x[:, t]))
            return torch.stack(outs, dim=1)
        if x.dim() == 4:
            return self._forward_single(x)
        raise ValueError(f"Expected BCHW or BTCHW input, got shape {tuple(x.shape)}")
