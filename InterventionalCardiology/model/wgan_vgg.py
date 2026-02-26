import torch
import torch.nn as nn


class WGANVGGGenerator(nn.Module):
    def __init__(
        self, in_channels: int = 1, base_channels: int = 32, num_blocks: int = 8
    ):
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(
                "WGAN-VGG generator baseline supports grayscale input (in_channels=1)."
            )
        c = int(base_channels)
        n = max(8, int(num_blocks))
        layers: list[nn.Module] = [nn.Conv2d(1, c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(2, n):
            layers.extend([nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(c, 1, 3, 1, 1), nn.ReLU(inplace=True)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, x.size(1) // 2]
        return self.net(x)


class WGANVGGDiscriminator(nn.Module):
    def __init__(self, input_size: int = 256, in_channels: int = 1):
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(
                "WGAN-VGG discriminator baseline supports grayscale input (in_channels=1)."
            )

        def _conv_out(n: int, k_list: list[int], s_list: list[int]) -> int:
            out = (n - k_list[0]) // s_list[0] + 1
            for k, s in zip(k_list[1:], s_list[1:]):
                out = (out - k) // s + 1
            return out

        layers: list[nn.Module] = []
        channels = [
            (1, 64, 1),
            (64, 64, 2),
            (64, 128, 1),
            (128, 128, 2),
            (128, 256, 1),
            (256, 256, 2),
        ]
        for ch_in, ch_out, stride in channels:
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.features = nn.Sequential(*layers)
        out_size = _conv_out(int(input_size), [3] * 6, [1, 2, 1, 2, 1, 2])
        if out_size <= 0:
            raise ValueError(f"Invalid discriminator input_size: {input_size}")
        self.fc1 = nn.Linear(256 * out_size * out_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, x.size(1) // 2]
        feat = self.features(x)
        feat = feat.reshape(feat.size(0), -1)
        feat = self.lrelu(self.fc1(feat))
        return self.fc2(feat)
