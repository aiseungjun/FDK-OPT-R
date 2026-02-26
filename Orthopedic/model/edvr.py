import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


def make_layer(block, num_blocks: int, **kwargs) -> nn.Sequential:
    layers = [block(**kwargs) for _ in range(int(num_blocks))]
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.relu(self.conv1(x)))
        return x + out


class DCNv2Pack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deformable_groups: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = int(groups)
        self.deformable_groups = int(deformable_groups)
        self.kh = int(kh)
        self.kw = int(kw)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // self.groups, self.kh, self.kw)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.conv_offset = nn.Conv2d(
            out_channels,
            self.deformable_groups * 3 * self.kh * self.kw,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class PCDAlignment(nn.Module):
    def __init__(self, num_feat: int = 64, deformable_groups: int = 8):
        super().__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        for i in range(3, 0, -1):
            level = f"l{i}"
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=1,
                deformable_groups=deformable_groups,
            )
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            stride=1,
            padding=1,
            deformable_groups=deformable_groups,
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(
        self, nbr_feat_l: list[torch.Tensor], ref_feat_l: list[torch.Tensor]
    ) -> torch.Tensor:
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f"l{i}"
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                if (
                    upsampled_offset is not None
                    and upsampled_offset.shape[-2:] != offset.shape[-2:]
                ):
                    upsampled_offset = F.interpolate(
                        upsampled_offset,
                        size=offset.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                offset = self.lrelu(
                    self.offset_conv2[level](
                        torch.cat([offset, upsampled_offset], dim=1)
                    )
                )
                offset = self.lrelu(self.offset_conv3[level](offset))
            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                if (
                    upsampled_feat is not None
                    and upsampled_feat.shape[-2:] != feat.shape[-2:]
                ):
                    upsampled_feat = F.interpolate(
                        upsampled_feat,
                        size=feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset)))
        )
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    def __init__(
        self, num_feat: int = 64, num_frame: int = 5, center_frame_idx: int = 2
    ):
        super().__init__()
        self.center_frame_idx = int(center_frame_idx)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, aligned_feat: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = aligned_feat.size()
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx].clone()
        )
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w)).view(
            b, t, -1, h, w
        )
        corr_l = []
        for i in range(t):
            corr = torch.sum(embedding[:, i] * embedding_ref, 1)
            corr_l.append(corr.unsqueeze(1))
        corr_prob = (
            torch.sigmoid(torch.cat(corr_l, dim=1)).unsqueeze(2).expand(b, t, c, h, w)
        )
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob.contiguous().view(
            b, -1, h, w
        )
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1))
        )
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)
        if attn_level.shape[-2:] != attn.shape[-2:]:
            attn_level = F.interpolate(
                attn_level, size=attn.shape[-2:], mode="bilinear", align_corners=False
            )
        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        if attn.shape[-2:] != feat.shape[-2:]:
            attn = F.interpolate(
                attn, size=feat.shape[-2:], mode="bilinear", align_corners=False
            )
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)
        feat = feat * attn * 2 + attn_add
        return feat


class EDVR(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_frames: int = 5,
        channels: int = 32,
        num_blocks: int = 8,
        deformable_groups: int = 8,
        with_tsa: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_frames = int(num_frames)
        self.center_frame_idx = self.num_frames // 2
        self.with_tsa = bool(with_tsa)

        num_feat = int(channels)
        self.conv_first = nn.Conv2d(self.in_channels, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.pcd_align = PCDAlignment(
            num_feat=num_feat, deformable_groups=deformable_groups
        )
        if self.with_tsa:
            self.fusion = TSAFusion(
                num_feat=num_feat,
                num_frame=self.num_frames,
                center_frame_idx=self.center_frame_idx,
            )
        else:
            self.fusion = nn.Conv2d(self.num_frames * num_feat, num_feat, 1, 1)
        self.reconstruction = make_layer(
            ResidualBlockNoBN, max(4, int(num_blocks)), num_feat=num_feat
        )
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, self.in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def _to_bcthw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            return x
        if x.dim() == 4:
            return x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
        raise ValueError(f"EDVR expects BCHW or BTCHW, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bcthw(x)
        b, t, c, h, w = x.size()
        if t != self.num_frames:
            raise ValueError(f"EDVR expects num_frames={self.num_frames}, got {t}")

        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        h2, w2 = feat_l2.shape[-2:]
        h3, w3 = feat_l3.shape[-2:]
        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h2, w2)
        feat_l3 = feat_l3.view(b, t, -1, h3, w3)

        ref_feat_l = [
            feat_l1[:, self.center_frame_idx].clone(),
            feat_l2[:, self.center_frame_idx].clone(),
            feat_l3[:, self.center_frame_idx].clone(),
        ]

        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [
                feat_l1[:, i].clone(),
                feat_l2[:, i].clone(),
                feat_l3[:, i].clone(),
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            feat = self.fusion(aligned_feat.view(b, -1, h, w))

        out = self.reconstruction(feat)
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = x[:, self.center_frame_idx]
        return torch.clamp(out + base, 0.0, 1.0)
