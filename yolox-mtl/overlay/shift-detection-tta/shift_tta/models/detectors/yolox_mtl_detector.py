"""YOLOX-MTL detector: mmdet YOLOX with an added semantic segmentation head.

Composition rather than inheritance so the mmdet YOLOX class stays untouched
and any future mmdet version bump does not break our subclass:

    x -> backbone -> neck (PAFPN) -> (feats)
                                 \\-> bbox_head  (existing YOLOX detection)
                                  \\-> seg_head  (our YOLOXSemSegHead)

Both heads share the same PAFPN features. The seg head is a small
Panoptic-FPN-style module (see YOLOXSemSegHead).

Training-time forward path
--------------------------
Inputs (via `AdaptiveDetector.loss` -> `loss(img, data_samples, **kw)`)::

    dict('loss_bbox': ..., 'loss_obj': ..., 'loss_cls': ..., 'loss_seg': ...)

The detection losses come from mmdet YOLOX's ``bbox_head.loss_by_feat`` and
the seg loss from ``YOLOXSemSegHead.loss``. Segmentation ground truth is read
from ``data_samples[i].gt_sem_seg.sem_seg`` (mmdet Cityscapes semseg
convention).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures import SampleList as DetSampleList

from shift_tta.registry import MODELS
from .base import BaseAdaptiveDetector


@MODELS.register_module()
class YOLOXMTLDetector(nn.Module):
    """YOLOX + semantic segmentation head sharing the same PAFPN.

    This is NOT a subclass of ``BaseAdaptiveDetector``. It is meant to be
    assigned to ``AdaptiveDetector.detector`` in the exact same way the
    single-task YOLOX detector is assigned in the published configs.

    Args:
        yolox: mmdet config dict for the YOLOX detector (``_scope_='mmdet'``,
            ``type='YOLOX'`` or ``type='mmdet.YOLOX'``). Same as your prior
            single-task config's ``detector`` field.
        seg_head: config dict for :class:`YOLOXSemSegHead`.
        seg_ignore_index: label value used to mark ignore in seg gt.
    """

    def __init__(
        self,
        yolox: dict,
        seg_head: dict,
        seg_ignore_index: int = 255,
    ):
        super().__init__()
        # The underlying mmdet YOLOX (backbone + neck + bbox_head).
        self.yolox = MMDET_MODELS.build(yolox)
        # Trigger the mmdet YOLOX's own init_cfg (Pretrained/COCO); the outer
        # runner's init_weights walk stops at us because we inherit from
        # nn.Module, not BaseModule.
        self.yolox.init_weights()
        # Our semantic segmentation head (registered in shift_tta.registry).
        self.seg_head = MODELS.build(seg_head)
        self.seg_ignore_index = int(seg_ignore_index)

    # ------------------------------------------------------------------
    # Backbone + neck -> shared PAFPN feats
    # ------------------------------------------------------------------
    def extract_feat(self, img: Tensor) -> Tuple[Tensor, ...]:
        """Shared trunk: backbone + neck. Returns (P3, P4, P5)."""
        return self.yolox.extract_feat(img)

    @property
    def bbox_head(self):
        # YOLOXModeSwitchHook accesses model.detector.bbox_head directly.
        return self.yolox.bbox_head

    # ------------------------------------------------------------------
    # Training-time loss
    # ------------------------------------------------------------------
    def loss(self, img: Tensor, data_samples: DetSampleList, **kwargs) -> Dict[str, Tensor]:
        feats = self.extract_feat(img)

        # Detection losses (mmdet YOLOX): reuse bbox_head.loss_by_feat via
        # the mmdet loss method to avoid duplicating that logic.
        outs = self.yolox.bbox_head(feats)
        det_losses = self.yolox.bbox_head.loss_by_feat(
            *outs, batch_gt_instances=[d.gt_instances for d in data_samples],
            batch_img_metas=[d.metainfo for d in data_samples],
            batch_gt_instances_ignore=[
                getattr(d, "ignored_instances", None) for d in data_samples
            ],
        )

        # Seg loss (only computed if seg gt is present in the batch)
        seg_targets = self._collect_seg_targets(data_samples, img.shape[-2:])
        if seg_targets is not None:
            seg_logits = self.seg_head(feats)
            det_losses["loss_seg"] = self.seg_head.loss(seg_logits, seg_targets)

        return det_losses

    # ------------------------------------------------------------------
    # Inference — mirrors mmdet YOLOX.predict but also returns seg logits.
    # ------------------------------------------------------------------
    def predict(
        self,
        img: Tensor,
        data_samples: DetSampleList,
        rescale: bool = True,
        **kwargs,
    ) -> DetSampleList:
        feats = self.extract_feat(img)
        # Detection path (unchanged mmdet YOLOX)
        outs = self.yolox.bbox_head(feats)
        det_predictions = self.yolox.bbox_head.predict_by_feat(
            *outs, batch_img_metas=[d.metainfo for d in data_samples],
            rescale=rescale,
        )
        data_samples = self.yolox.bbox_head.add_pred_to_datasample(
            data_samples, det_predictions
        ) if hasattr(self.yolox.bbox_head, "add_pred_to_datasample") else \
            self.yolox.add_pred_to_datasample(data_samples, det_predictions)

        # Seg path -- upsample logits to input res and attach argmax to samples
        seg_logits = self.seg_head(feats)
        input_hw = img.shape[-2:]
        seg_pred = self.seg_head.predict(seg_logits, input_hw)
        for i, d in enumerate(data_samples):
            d.pred_sem_seg = _make_pixel_data(seg_pred[i])
            d.seg_logits = seg_logits[i].detach()

        return data_samples

    # ------------------------------------------------------------------
    # Convenience: returns raw feats + head outputs (used by the adapter)
    # ------------------------------------------------------------------
    def forward_train_outputs(self, img: Tensor) -> Dict[str, Tensor]:
        """Return backbone-neck features and both heads' raw outputs.

        Used by the CT-CMT adapter to compute consistency and contrastive
        losses without going through the full mmdet loss machinery.
        """
        feats = self.extract_feat(img)
        det_outs = self.yolox.bbox_head(feats)  # (cls, reg, obj) triples per level
        seg_logits = self.seg_head(feats)
        return {"feats": feats, "det_outs": det_outs, "seg_logits": seg_logits}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _collect_seg_targets(
        self,
        data_samples: DetSampleList,
        input_hw: Tuple[int, int],
    ) -> Optional[Tensor]:
        """Stack per-sample seg gt into (B, H, W) long. Returns None if any
        sample is missing seg gt (skips the seg loss for that batch)."""
        targets = []
        H_in, W_in = input_hw
        for d in data_samples:
            gt = getattr(d, "gt_sem_seg", None)
            if gt is None:
                return None
            m = gt.sem_seg  # (1, H, W)
            if m.dim() == 3:
                m = m[0]
            if m.shape[-2:] != (H_in, W_in):
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=(H_in, W_in), mode="nearest",
                ).squeeze(0).squeeze(0)
            targets.append(m)
        return torch.stack([t.long() for t in targets], dim=0)


def _make_pixel_data(sem_seg: Tensor):
    """Wrap a (H, W) tensor into an mmengine PixelData; fall back to a bare
    ``types.SimpleNamespace`` if mmengine is not importable at runtime."""
    try:
        from mmengine.structures import PixelData
        return PixelData(sem_seg=sem_seg[None])
    except Exception:
        from types import SimpleNamespace
        return SimpleNamespace(sem_seg=sem_seg[None])
