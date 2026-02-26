import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        depth = max(2, int(depth))
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = self.in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.downs.append(_DoubleConv(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        self.bottleneck = _DoubleConv(ch, ch)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in reversed(range(depth)):
            in_ch = base_channels * (2**i)
            self.ups.append(nn.ConvTranspose2d(ch, in_ch, kernel_size=2, stride=2))
            self.up_convs.append(_DoubleConv(in_ch * 2, in_ch))
            ch = in_ch

        self.final = nn.Conv2d(ch, self.out_channels, kernel_size=1)
        self.skip = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.skip(x_in) + self.final(x)
