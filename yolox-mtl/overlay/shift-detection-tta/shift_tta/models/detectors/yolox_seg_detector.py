"""YOLOX-Seg detector: same CSPDarknet + YOLOXPAFPN trunk as YOLOX, but only
the semantic segmentation head — no bbox head.

Used as the single-task segmentation baseline in the MTL comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures import SampleList as DetSampleList
from mmengine.structures import InstanceData

from shift_tta.registry import MODELS


def _make_pixel_data(sem_seg: Tensor):
    try:
        from mmengine.structures import PixelData
        return PixelData(sem_seg=sem_seg[None])
    except Exception:
        from types import SimpleNamespace
        return SimpleNamespace(sem_seg=sem_seg[None])


@MODELS.register_module()
class YOLOXSegDetector(nn.Module):
    """Backbone + neck + YOLOXSemSegHead. No detection head.

    Args:
        backbone: mmdet-style backbone config (usually ``CSPDarknet``).
        neck:     mmdet-style neck config (usually ``YOLOXPAFPN``).
        seg_head: :class:`YOLOXSemSegHead` config.
        seg_ignore_index: label value used to mark ignore in seg gt.
        init_cfg: optional mmengine init_cfg (e.g. ``Pretrained`` for YOLOX
            COCO weights — only backbone+neck params will match).
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        seg_head: dict,
        seg_ignore_index: int = 255,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone = MMDET_MODELS.build(backbone)
        self.neck = MMDET_MODELS.build(neck)
        self.seg_head = MODELS.build(seg_head)
        self.seg_ignore_index = int(seg_ignore_index)

        if init_cfg is not None:
            from mmengine.runner import load_checkpoint
            if init_cfg.get("type") == "Pretrained":
                # Best-effort partial load: only backbone/neck weights will match.
                load_checkpoint(self, init_cfg["checkpoint"],
                                map_location="cpu", strict=False)

    def extract_feat(self, img: Tensor) -> Tuple[Tensor, ...]:
        return self.neck(self.backbone(img))

    def loss(self, img: Tensor, data_samples: DetSampleList,
             **kwargs) -> Dict[str, Tensor]:
        feats = self.extract_feat(img)
        seg_logits = self.seg_head(feats)
        seg_targets = self._collect_seg_targets(data_samples, img.shape[-2:])
        losses: Dict[str, Tensor] = {}
        if seg_targets is not None:
            losses["loss_seg"] = self.seg_head.loss(seg_logits, seg_targets)
        return losses

    def predict(self, img: Tensor, data_samples: DetSampleList,
                rescale: bool = True, **kwargs) -> DetSampleList:
        feats = self.extract_feat(img)
        seg_logits = self.seg_head(feats)
        input_hw = img.shape[-2:]
        seg_pred = self.seg_head.predict(seg_logits, input_hw)  # (B, H, W)

        empty_inst = InstanceData(
            bboxes=seg_logits.new_zeros((0, 4)),
            labels=seg_logits.new_zeros((0,), dtype=torch.long),
            scores=seg_logits.new_zeros((0,)),
        )
        for i, d in enumerate(data_samples):
            d.pred_sem_seg = _make_pixel_data(seg_pred[i])
            # AdaptiveDetector.predict expects pred_instances; keep empty.
            d.pred_instances = empty_inst
        return data_samples

    def _collect_seg_targets(
        self,
        data_samples: DetSampleList,
        input_hw: Tuple[int, int],
    ) -> Optional[Tensor]:
        targets: List[Tensor] = []
        H_in, W_in = input_hw
        for d in data_samples:
            gt = getattr(d, "gt_sem_seg", None)
            if gt is None:
                return None
            m = gt.sem_seg
            if m.dim() == 3:
                m = m[0]
            if m.shape[-2:] != (H_in, W_in):
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=(H_in, W_in), mode="nearest",
                ).squeeze(0).squeeze(0)
            targets.append(m)
        return torch.stack([t.long() for t in targets], dim=0)
