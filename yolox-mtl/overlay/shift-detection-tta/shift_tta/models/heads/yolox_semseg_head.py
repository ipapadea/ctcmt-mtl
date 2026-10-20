"""Panoptic-FPN-style semantic segmentation head fed by YOLOX's PAFPN.

Takes the 3 PAFPN feature maps at strides 8/16/32, upsamples P4/P5 to the
P3 grid, sums (or concatenates) them into a single stride-8 feature, and
runs a small conv decoder to produce (B, num_classes, H/8, W/8) logits.
During loss/eval the logits are bilinearly upsampled to the input resolution.

This module is intentionally lightweight relative to YOLOX itself:
  - default hidden_channels=128
  - default 2 conv blocks per level

so that seg does not dominate the parameter count or gradient scale of the
detector during MTL training.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shift_tta.registry import MODELS


class _ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@MODELS.register_module()
class YOLOXSemSegHead(nn.Module):
    """Semantic segmentation head consuming YOLOX PAFPN outputs.

    Args:
        in_channels: channels of the (P3, P4, P5) PAFPN outputs. YOLOX-X uses
            (320, 320, 320); YOLOX-S uses (128, 128, 128). Pass the same triple.
        num_classes: number of semantic-seg classes. Cityscapes: 19.
        hidden_channels: internal decoder width. Default 128.
        num_convs_per_level: how many ConvBNReLU per FPN level before fusion.
        fusion: ``'add'`` or ``'cat'``. Default ``'add'`` (memory-lighter).
        ignore_index: ignore label for the training loss. Default 255.
        loss_weight: multiplier for the seg CE loss reported in ``loss(...)``.
        dropout_p: optional 2d dropout right before the classifier. 0.1 helps
            regularize on small datasets like Cityscapes.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (320, 320, 320),
        num_classes: int = 19,
        hidden_channels: int = 128,
        num_convs_per_level: int = 2,
        fusion: str = "add",
        ignore_index: int = 255,
        loss_weight: float = 1.0,
        dropout_p: float = 0.1,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(
                f"in_channels must have 3 entries (P3, P4, P5), got {in_channels}"
            )
        if fusion not in ("add", "cat"):
            raise ValueError(f"fusion must be 'add' or 'cat', got {fusion}")

        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.loss_weight = float(loss_weight)
        self.fusion = fusion

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        # Per-level projection + refinement to `hidden_channels`
        self.level_convs = nn.ModuleList()
        for c_in in in_channels:
            layers = [_ConvBNReLU(c_in, hidden_channels, 1)]
            for _ in range(num_convs_per_level - 1):
                layers.append(_ConvBNReLU(hidden_channels, hidden_channels, 3))
            self.level_convs.append(nn.Sequential(*layers))

        fused_c = (
            hidden_channels if fusion == "add" else hidden_channels * 3
        )
        self.fuse = _ConvBNReLU(fused_c, hidden_channels, 3)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)

    # ------------------------------------------------------------------
    def _upsample_to(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats: 3 tensors from YOLOX PAFPN at strides 8/16/32, ordered
                (P3, P4, P5) — same order YOLOX bbox head consumes.
        Returns:
            Logits at the (P3 grid) stride 8 resolution: (B, C, H/8, W/8).
            Upsample to input elsewhere (during loss + eval).
        """
        if len(feats) != 3:
            raise RuntimeError(
                f"expected 3 PAFPN feats, got {len(feats)}"
            )
        p3, p4, p5 = feats
        target_hw = p3.shape[-2:]

        y3 = self.level_convs[0](p3)
        y4 = self._upsample_to(self.level_convs[1](p4), target_hw)
        y5 = self._upsample_to(self.level_convs[2](p5), target_hw)

        if self.fusion == "add":
            fused = y3 + y4 + y5
        else:
            fused = torch.cat([y3, y4, y5], dim=1)

        fused = self.fuse(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

    # ------------------------------------------------------------------
    def loss(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss (ignore_index-aware, optional class_weights).
        Upsamples logits to the target (input) resolution so boundary
        supervision is not lost by downsampling the GT masks."""
        if seg_logits.shape[-2:] != seg_targets.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=seg_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.loss_weight * F.cross_entropy(
            seg_logits,
            seg_targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )

    # ------------------------------------------------------------------
    def predict(
        self,
        seg_logits: torch.Tensor,
        input_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Upsample seg logits to input resolution and return argmax
        (B, H_in, W_in) int64. Handy for eval-time visualization."""
        up = F.interpolate(seg_logits, size=input_hw, mode="bilinear",
                           align_corners=False)
        return up.argmax(dim=1)
