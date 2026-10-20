"""Cross-Task Contrastive Learning (CT-CL) loss for CT-CMT.

Extends the published object-level supervised contrastive loss (Moraiti et al.
2026) to include a *segmentation view* of the same object alongside the
existing *detection view*.

For each teacher-detected bbox ``b`` with class ``c``:

    z_det(b) = RoIAlign(feature_map, b)                     # existing view
    z_seg(b) = mask-weighted average of feature_map inside b,
               weighted by teacher_seg_probs[..., c][b]     # NEW view

The two views are L2-normalized and fed to :class:`SupConLoss` (unchanged
implementation reused from the prior work) with class labels ``c``. Same-class
pairs (including det<->seg pairs of the same object) are pulled together;
different-class pairs are pushed apart. This creates a shared cross-task
representation space.

Single-task fallback
--------------------
If no seg feature maps / seg probabilities are supplied, only the det view
contributes and the loss degenerates to the prior published SupCon setup.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from shift_tta.registry import MODELS
from .supervised_contrastive_loss import SupConLoss


@MODELS.register_module()
class CrossTaskContrastiveLoss(nn.Module):
    """CT-CL: cross-task supervised contrastive over det+seg feature views.

    Args:
        weight: overall loss weight.
        temperature: SupCon temperature. Default 0.07 (matches prior work).
        roi_output_size: RoIAlign spatial output. Default (7, 7).
        num_scales: number of feature-map scales expected in ``feature_maps``.
            Loss is computed independently at each scale and summed. Default 3
            (matches YOLOX PAFPN P3/P4/P5). Only used to guard input shape.
        min_positives: minimum number of positive pairs (excluding self) needed
            for a scale's loss to be computed. Prevents SupCon numerical issues
            when a class appears only once in the batch.
        include_seg_view: master switch. If False, only det features are used
            (equivalent to the prior single-task supervised contrastive setup).
        seg_view_softmax_prior: teacher probabilities are optionally
            renormalized inside each bbox before being used as weights,
            preventing background pixels from washing out the aggregation.
    """

    def __init__(
        self,
        weight: float = 0.01,
        temperature: float = 0.07,
        roi_output_size: Tuple[int, int] = (7, 7),
        num_scales: int = 3,
        min_positives: int = 1,
        include_seg_view: bool = True,
        seg_view_softmax_prior: bool = True,
    ):
        super().__init__()
        self.weight = float(weight)
        self.roi_output_size = tuple(roi_output_size)
        self.num_scales = int(num_scales)
        self.min_positives = int(min_positives)
        self.include_seg_view = bool(include_seg_view)
        self.seg_view_softmax_prior = bool(seg_view_softmax_prior)

        self.supcon = SupConLoss(
            temperature=temperature,
            contrast_mode="all",
            base_temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_rois(
        boxes_per_image: List[torch.Tensor],
    ) -> torch.Tensor:
        """Convert a list of (N_i, 4) xyxy boxes into torchvision's
        (K, 5) ``[batch_idx, x1, y1, x2, y2]`` tensor."""
        packed = []
        for i, boxes in enumerate(boxes_per_image):
            if boxes.numel() == 0:
                continue
            idx = torch.full((boxes.size(0), 1), float(i),
                             device=boxes.device, dtype=boxes.dtype)
            packed.append(torch.cat([idx, boxes], dim=1))
        if not packed:
            return torch.zeros((0, 5))
        return torch.cat(packed, dim=0)

    def _det_view_features(
        self,
        feature_map: torch.Tensor,
        rois: torch.Tensor,
        spatial_scale: float,
    ) -> torch.Tensor:
        """Standard RoIAlign at bbox regions -> global-average per RoI ->
        L2-normalized (B_total, C)."""
        if rois.numel() == 0:
            return torch.zeros((0, feature_map.size(1)), device=feature_map.device)
        pooled = roi_align(
            feature_map, rois.to(feature_map.dtype),
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=2, aligned=True,
        )  # (K, C, oh, ow)
        pooled = pooled.mean(dim=(2, 3))
        return F.normalize(pooled, p=2, dim=1, eps=1e-12)

    def _seg_view_features(
        self,
        feature_map: torch.Tensor,
        boxes_per_image: List[torch.Tensor],
        classes_per_image: List[torch.Tensor],
        teacher_seg_probs: torch.Tensor,
        spatial_scale: float,
    ) -> torch.Tensor:
        """For each bbox b with class c, aggregate feature_map inside b using
        teacher_seg_probs[..., c][b] as weights. Returns (B_total, C)."""
        if not any(bo.numel() for bo in boxes_per_image):
            return torch.zeros((0, feature_map.size(1)), device=feature_map.device)

        Bf, C, Hf, Wf = feature_map.shape
        _, K_seg, Hs, Ws = teacher_seg_probs.shape

        # Bring seg probs onto the feature-map grid so weighting is aligned.
        if (Hs, Ws) != (Hf, Wf):
            teacher_seg_probs = F.interpolate(
                teacher_seg_probs, size=(Hf, Wf),
                mode="bilinear", align_corners=False,
            )

        results = []
        for i, (boxes, classes) in enumerate(zip(boxes_per_image, classes_per_image)):
            if boxes.numel() == 0:
                continue
            fm_i = feature_map[i]                 # (C, Hf, Wf)
            probs_i = teacher_seg_probs[i]        # (K_seg, Hf, Wf)

            # Convert xyxy boxes from *input* pixel space to feature-map pixel
            # space via spatial_scale (same convention as RoIAlign).
            b_fs = boxes.to(fm_i.dtype) * float(spatial_scale)
            b_fs[:, 0::2] = b_fs[:, 0::2].clamp(0, Wf)
            b_fs[:, 1::2] = b_fs[:, 1::2].clamp(0, Hf)

            for (x1, y1, x2, y2), cls_idx in zip(b_fs.tolist(), classes.tolist()):
                x1i, y1i = int(x1), int(y1)
                x2i, y2i = max(int(x2), x1i + 1), max(int(y2), y1i + 1)
                if x2i - x1i < 1 or y2i - y1i < 1:
                    results.append(torch.zeros(C, device=fm_i.device))
                    continue
                cls_idx = int(cls_idx)
                if cls_idx < 0 or cls_idx >= K_seg:
                    # class outside seg taxonomy -> fall back to plain average
                    weights = torch.ones(
                        (y2i - y1i, x2i - x1i), device=fm_i.device
                    )
                else:
                    weights = probs_i[cls_idx, y1i:y2i, x1i:x2i]

                if self.seg_view_softmax_prior:
                    weights = weights + 1e-8
                    weights = weights / weights.sum()

                patch = fm_i[:, y1i:y2i, x1i:x2i]           # (C, h, w)
                weighted = (patch * weights.unsqueeze(0)).sum(dim=(1, 2))
                results.append(weighted)

        if not results:
            return torch.zeros((0, feature_map.size(1)), device=feature_map.device)
        z = torch.stack(results, dim=0)
        return F.normalize(z, p=2, dim=1, eps=1e-12)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        feature_maps: Sequence[torch.Tensor],
        boxes_per_image: List[torch.Tensor],
        classes_per_image: List[torch.Tensor],
        spatial_scales: Sequence[float],
        teacher_seg_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feature_maps: sequence of teacher feature maps at different
                scales (e.g. YOLOX PAFPN P3/P4/P5). Same batch size across all.
            boxes_per_image: length-B list, each (N_i, 4) xyxy in *input*
                pixel space (matches how boxes are stored in TrackDataSample).
            classes_per_image: length-B list, each (N_i,) long, class ids
                aligned with ``boxes_per_image`` and with ``teacher_seg_probs``
                channel indices.
            spatial_scales: one value per feature map, mapping input pixel
                coord -> feature-map coord (e.g. 1/8, 1/16, 1/32).
            teacher_seg_probs: (B, K_seg, Hs, Ws) softmax probs from the
                teacher's seg head. Required iff ``include_seg_view=True``
                and there are any boxes.

        Returns:
            Scalar loss (already multiplied by ``self.weight``). Zero if no
            usable positive pairs are present at any scale.
        """
        if len(feature_maps) != len(spatial_scales):
            raise ValueError(
                f"feature_maps ({len(feature_maps)}) and spatial_scales "
                f"({len(spatial_scales)}) must match"
            )

        device = feature_maps[0].device
        n_boxes_total = sum(int(b.size(0)) for b in boxes_per_image)
        if n_boxes_total == 0:
            return torch.zeros((), device=device)

        rois = self._pack_rois(boxes_per_image).to(device)
        flat_classes = torch.cat(
            [c for c in classes_per_image if c.numel() > 0], dim=0
        ).long().to(device)

        need_seg = self.include_seg_view and teacher_seg_probs is not None
        total = torch.zeros((), device=device)

        for fmap, spatial_scale in zip(feature_maps, spatial_scales):
            z_det = self._det_view_features(fmap, rois, spatial_scale)

            if need_seg:
                z_seg = self._seg_view_features(
                    fmap, boxes_per_image, classes_per_image,
                    teacher_seg_probs, spatial_scale,
                )
                features = torch.cat([z_det.unsqueeze(1), z_seg.unsqueeze(1)], dim=1)
                labels = flat_classes
            else:
                # Only det view -> match prior work's shape [K, 1, C]
                features = z_det.unsqueeze(1)
                labels = flat_classes

            n_positives = _count_positive_pairs(labels)
            if n_positives < self.min_positives:
                continue

            total = total + self.supcon(features, labels=labels)

        return self.weight * total


def _count_positive_pairs(labels: torch.Tensor) -> int:
    """Number of same-class pairs (excluding self-pairs). Used to skip
    SupCon when the batch has no positives at all -> the loss would just be
    the log-partition-function estimate, which is unstable and uninformative."""
    if labels.numel() < 2:
        return 0
    eq = (labels.view(-1, 1) == labels.view(1, -1)).float()
    eq.fill_diagonal_(0)
    return int(eq.sum().item())
