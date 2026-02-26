import torch
import torch.nn as nn
import torch.nn.functional as F


class REDCNN(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 96, num_layers: int = 5):
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(
                "REDCNN baseline supports grayscale input (in_channels=1)."
            )
        if int(num_layers) != 5:
            raise ValueError("REDCNN reference architecture uses num_layers=5.")

        c = int(channels)
        self.conv1 = nn.Conv2d(1, c, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(c, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, x.size(1) // 2]
        if x.dim() != 4:
            raise ValueError(f"REDCNN expects BCHW/BTCHW, got {tuple(x.shape)}")

        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))

        out = self.tconv1(out)
        out = out + residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out = out + residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out = out + residual_1
        return torch.clamp(self.relu(out), 0.0, 1.0)
