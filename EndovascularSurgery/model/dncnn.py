import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels: int = 1, depth: int = 12, features: int = 64):
        super().__init__()
        depth = max(4, int(depth))
        features = int(features)

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(features, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, x.size(1) // 2]
        noise = self.net(x)
        return torch.clamp(x - noise, 0.0, 1.0)
