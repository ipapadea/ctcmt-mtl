"""Single-branch port of MultiHGDModuleTwinLiteNetv2Scaled.

Codebook-attention decoder that turns three multi-scale feature maps into
one high-resolution feature map. Ported from TwinLiteNet v2 with the
``lane`` output branch removed (we only have one semantic-seg task).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(c_in: int, c_out: int, norm_layer) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 1, bias=False),
        norm_layer(c_out),
        nn.ReLU(inplace=True),
    )


class MultiHGDModule(nn.Module):
    """Multi-scale Hierarchical Guided Decoder (single-branch variant).

    Inputs:
        x       : (N, C, H,  W )  low-res feature map (stride 32).
        guide1  : (N, C, H1, W1)  mid-res feature map (stride 16).
        guide2  : (N, C, H2, W2)  high-res feature map (stride 8).
        All three must share ``in_channels`` (typically produced by 1x1
        channel-reducer convs upstream).

    Output:
        (N, out_channels, H2, W2) fused feature map at the high-res grid.

    Args:
        in_channels: channels of the three input maps after upstream reduction.
        center_channels: codebook width (# soft "centres" in the spatial pool).
        out_channels: decoder / classifier feature width.
        norm_layer: normalization to use (default: BatchNorm2d).
    """

    def __init__(
        self,
        in_channels: int,
        center_channels: int,
        out_channels: int,
        norm_layer: Optional[type] = None,
    ) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.in_channels = int(in_channels)
        self.center_channels = int(center_channels)
        self.out_channels = int(out_channels)

        # Low-res codebook builders (str-32 grid)
        self.conv_cat = _conv_bn_relu(in_channels * 3, out_channels, norm_layer)
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels * 3, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
        )
        self.norm_center = nn.Softmax(dim=2)

        # High-res affinity (single branch)
        self.conv_affinity0 = _conv_bn_relu(in_channels * 3, out_channels, norm_layer)
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_up = _conv_bn_relu(2 * out_channels, out_channels, norm_layer)
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        x: torch.Tensor,
        guide1: torch.Tensor,
        guide2: torch.Tensor,
    ) -> torch.Tensor:
        n, _, h, w = x.size()
        _, _, h2, w2 = guide2.size()

        x_up0 = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode="bilinear", align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode="bilinear", align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode="bilinear", align_corners=True)

        # ---- Codebook @ str-32 --------------------------------------------
        x_cat = torch.cat([guide2_down, guide1_down, x], dim=1)
        f_cat = self.conv_cat(x_cat)                             # (N, out, h, w)
        f_center = self.conv_center(x_cat)                       # (N, center, h, w)

        f_cat_flat = f_cat.view(n, self.out_channels, h * w)     # (N, out, hw)
        f_center_flat = self.norm_center(
            f_center.view(n, self.center_channels, h * w)
        )                                                        # softmax over hw
        x_center = f_cat_flat.bmm(f_center_flat.transpose(1, 2)) # (N, out, center)

        f_cat_avg = self.avgpool0(f_cat).repeat(1, 1, h2, w2)    # (N, out, h2, w2)

        # ---- Reproject codebook @ str-8 -----------------------------------
        guide_cat = torch.cat([guide2, x_up1, x_up0], dim=1)
        aff0 = self.conv_affinity0(guide_cat)                    # (N, out, h2, w2)
        aff = self.conv_affinity1(aff0 + f_cat_avg)              # (N, center, h2, w2)
        aff_flat = aff.view(n, self.center_channels, h2 * w2)

        norm = self.center_channels ** -0.5
        x_up = norm * x_center.bmm(aff_flat)                     # (N, out, h2 w2)
        x_up = x_up.view(n, self.out_channels, h2, w2)

        return self.conv_up(torch.cat([x_up, aff0], dim=1))
