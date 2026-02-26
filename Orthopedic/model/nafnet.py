import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ):
        ctx.eps = eps
        n, c, h, w = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        eps = ctx.eps
        n, c, h, w = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (1.0 / torch.sqrt(var + eps)) * (g - y * mean_gy - mean_g)
        gw = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        gb = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)
        return gx, gw, gb, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        c: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(
            c, dw_channel, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            dw_channel // 2, c, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        self.sg = SimpleGate()
        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(
            c, ffn_channel, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.conv5 = nn.Conv2d(
            ffn_channel // 2, c, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        blocks: int = 6,
    ):
        super().__init__()
        img_channel = int(in_channels)
        self.intro = nn.Conv2d(
            img_channel, width, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.ending = nn.Conv2d(
            width, img_channel, kernel_size=3, stride=1, padding=1, bias=True
        )

        enc_blk_nums = [max(1, int(blocks // 3))] * 3
        dec_blk_nums = [max(1, int(blocks // 3))] * 3
        middle_blk_num = max(1, int(blocks // 2))

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = int(width)
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def _check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.dim() == 5:
            inp = inp[:, inp.size(1) // 2]
        b, c, h, w = inp.shape
        x = self._check_image_size(inp)
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + self._check_image_size(inp)
        x = x[:, :, :h, :w]
        return torch.clamp(x, 0.0, 1.0)
