import torch
import torch.nn as nn


def _ensure_bcthw(
    x: torch.Tensor, in_channels: int, num_frames: int = 5
) -> torch.Tensor:
    if x.dim() == 5:
        return x
    if x.dim() != 4:
        raise ValueError(f"FastDVDnet expects BCHW or BTCHW, got {tuple(x.shape)}")
    b, c, h, w = x.shape
    exp = in_channels * num_frames
    if c != exp:
        raise ValueError(f"4D FastDVDnet input must have {exp} channels, got {c}")
    return x.view(b, num_frames, in_channels, h, w)


class _CvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _InputCvBlock(nn.Module):
    def __init__(self, num_in_frames: int, in_channels: int, out_ch: int):
        super().__init__()
        self.interm_ch = 30
        self.block = nn.Sequential(
            nn.Conv2d(
                num_in_frames * (in_channels + 1),
                num_in_frames * self.interm_ch,
                kernel_size=3,
                padding=1,
                groups=num_in_frames,
                bias=False,
            ),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_in_frames * self.interm_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            _CvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            _CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _OutputCvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DenBlock(nn.Module):
    def __init__(
        self, in_channels: int, num_input_frames: int = 3, base_channels: int = 32
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.chs_lyr0 = int(base_channels)
        self.chs_lyr1 = int(base_channels) * 2
        self.chs_lyr2 = int(base_channels) * 4
        self.inc = _InputCvBlock(
            num_input_frames, in_channels=self.in_channels, out_ch=self.chs_lyr0
        )
        self.downc0 = _DownBlock(self.chs_lyr0, self.chs_lyr1)
        self.downc1 = _DownBlock(self.chs_lyr1, self.chs_lyr2)
        self.upc2 = _UpBlock(self.chs_lyr2, self.chs_lyr1)
        self.upc1 = _UpBlock(self.chs_lyr1, self.chs_lyr0)
        self.outc = _OutputCvBlock(self.chs_lyr0, self.in_channels)
        self.reset_params()

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def reset_params(self):
        for _name, m in self.named_modules():
            self._weight_init(m)

    def forward(
        self,
        in0: torch.Tensor,
        in1: torch.Tensor,
        in2: torch.Tensor,
        noise_map: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.inc(
            torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1)
        )
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x2 = self.upc2(x2)
        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = nn.functional.interpolate(
                x2, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )
        x1 = self.upc1(x1 + x2)
        if x1.shape[-2:] != x0.shape[-2:]:
            x1 = nn.functional.interpolate(
                x1, size=x0.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.outc(x0 + x1)
        if x.shape[-2:] != in1.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=in1.shape[-2:], mode="bilinear", align_corners=False
            )
        x = in1 - x
        return x


class FastDVDnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_frames: int = 5,
        default_noise_level: float = 0.03,
        base_channels: int = 32,
    ):
        super().__init__()
        if int(num_frames) != 5:
            raise ValueError("FastDVDnet is defined for 5-frame input.")
        self.in_channels = int(in_channels)
        self.num_frames = int(num_frames)
        self.default_noise_level = float(default_noise_level)
        self.base_channels = int(base_channels)
        self.temp1 = _DenBlock(
            in_channels=self.in_channels,
            num_input_frames=3,
            base_channels=self.base_channels,
        )
        self.temp2 = _DenBlock(
            in_channels=self.in_channels,
            num_input_frames=3,
            base_channels=self.base_channels,
        )
        self.reset_params()

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def reset_params(self):
        for _name, m in self.named_modules():
            self._weight_init(m)

    def _noise_map(
        self, x: torch.Tensor, noise_map: torch.Tensor | None
    ) -> torch.Tensor:
        b, _t, _c, h, w = x.shape
        if noise_map is None:
            return torch.full(
                (b, 1, h, w), self.default_noise_level, dtype=x.dtype, device=x.device
            )
        if noise_map.dim() == 4 and noise_map.size(1) == 1:
            return noise_map.to(device=x.device, dtype=x.dtype)
        raise ValueError("noise_map must be [B,1,H,W]")

    def forward(
        self, x: torch.Tensor, noise_map: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq = _ensure_bcthw(x, in_channels=self.in_channels, num_frames=self.num_frames)
        nm = self._noise_map(seq, noise_map)
        x0, x1, x2, x3, x4 = [seq[:, i] for i in range(5)]
        x20 = self.temp1(x0, x1, x2, nm)
        x21 = self.temp1(x1, x2, x3, nm)
        x22 = self.temp1(x2, x3, x4, nm)
        out = self.temp2(x20, x21, x22, nm)
        return torch.clamp(out, 0.0, 1.0)
