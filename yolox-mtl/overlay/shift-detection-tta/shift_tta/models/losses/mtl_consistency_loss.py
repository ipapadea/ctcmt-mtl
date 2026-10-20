"""MTL consistency loss for the CT-CMT adapter.

Extends :class:`YOLOXConsistencyContrastiveLoss` from the published prior work
(Moraiti et al. 2026) with a segmentation branch: teacher's seg softmax is
used as a soft pseudo-label to supervise the student's seg logits via
soft cross-entropy (matches CoTTA's ``softmax_entropy`` for seg).

Only the consistency terms live here; cross-task contrastive learning is a
separate module (:class:`CrossTaskContrastiveLoss`).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from shift_tta.registry import MODELS


@MODELS.register_module()
class MTLConsistencyLoss(nn.Module):
    """Multi-task consistency loss = YOLOX det MSE + seg soft-CE.

    Args:
        weight_det: multiplier for the aggregated detection consistency term.
            Matches the ``weight_consistency_loss`` of your prior single-task
            loss so ablation defaults transfer directly.
        weight_seg: multiplier for the seg soft-CE consistency.
        obj_weight, reg_weight, cls_weight: sub-weights inside the det term.
        seg_temperature: teacher softmax temperature; T>1 softens the target
            distribution (a mild TTA regularizer). Default 1.0 (paper spec).
    """

    def __init__(
        self,
        weight_det: float = 0.01,
        weight_seg: float = 1.0,
        obj_weight: float = 1.0,
        reg_weight: float = 1.0,
        cls_weight: float = 1.0,
        seg_temperature: float = 1.0,
    ):
        super().__init__()
        self.weight_det = float(weight_det)
        self.weight_seg = float(weight_seg)
        self.obj_weight = float(obj_weight)
        self.reg_weight = float(reg_weight)
        self.cls_weight = float(cls_weight)
        self.seg_temperature = float(seg_temperature)

    # ------------------------------------------------------------------
    def _yolox_det_consistency(
        self,
        student_outs: Dict[str, list],
        teacher_outs: Dict[str, list],
    ) -> torch.Tensor:
        """Per-level MSE consistency on obj, reg, cls.

        Follows :class:`YOLOXConsistencyContrastiveLoss` from the published
        prior work: elementwise MSE, averaged over batch and grid, then a
        weighted sum of the three subterms.
        """
        t_obj = teacher_outs["objectness"]
        t_reg = teacher_outs["bbox_pred"]
        t_cls = teacher_outs["cls_score"]
        s_obj = student_outs["objectness"]
        s_reg = student_outs["bbox_pred"]
        s_cls = student_outs["cls_score"]

        obj_loss = torch.zeros((), device=t_obj[0].device)
        reg_loss = torch.zeros((), device=t_obj[0].device)
        cls_loss = torch.zeros((), device=t_obj[0].device)

        for so, sr, sc, to_, tr, tc in zip(
            s_obj, s_reg, s_cls, t_obj, t_reg, t_cls,
        ):
            obj_loss = obj_loss + F.mse_loss(so, to_.detach())
            reg_loss = reg_loss + F.mse_loss(sr, tr.detach())
            cls_loss = cls_loss + F.mse_loss(sc, tc.detach())

        return (
            self.obj_weight * obj_loss
            + self.reg_weight * reg_loss
            + self.cls_weight * cls_loss
        )

    # ------------------------------------------------------------------
    def _seg_soft_ce(
        self,
        student_seg_logits: torch.Tensor,
        teacher_seg_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Soft cross-entropy between teacher softmax and student log-softmax.
        Both tensors are (B, C, H, W) at the same spatial resolution."""
        if student_seg_logits.shape != teacher_seg_logits.shape:
            teacher_seg_logits = F.interpolate(
                teacher_seg_logits,
                size=student_seg_logits.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        T = self.seg_temperature
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_seg_logits / T, dim=1)
        student_log_probs = F.log_softmax(student_seg_logits.float(), dim=1)
        # Mean over pixels and batch; sum over classes.
        return -(teacher_probs * student_log_probs).sum(dim=1).mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        student_det_outs: Dict[str, list],
        teacher_det_outs: Dict[str, list],
        student_seg_logits: Optional[torch.Tensor] = None,
        teacher_seg_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return a dict with the individual components and the total.

        ``student_seg_logits`` / ``teacher_seg_logits`` may be None -- in
        which case only the det term is computed (single-task fallback).
        """
        det_term = self._yolox_det_consistency(student_det_outs, teacher_det_outs)
        weighted_det = self.weight_det * det_term

        components = {"det": weighted_det}
        total = weighted_det

        if student_seg_logits is not None and teacher_seg_logits is not None:
            seg_term = self._seg_soft_ce(student_seg_logits, teacher_seg_logits)
            weighted_seg = self.weight_seg * seg_term
            components["seg"] = weighted_seg
            total = total + weighted_seg

        components["total"] = total
        return components
