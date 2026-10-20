"""Minimal mean-IoU metric for semantic segmentation.

Compatible with any ``BaseModel`` whose ``predict`` attaches
``data_sample.pred_sem_seg`` (a ``PixelData`` with attribute ``sem_seg`` of
shape ``(1, H, W)`` int64) and whose data-loading pipeline attaches
``data_sample.gt_sem_seg`` in the same format.

Ignore label defaults to 255 (Cityscapes trainIds convention).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from shift_tta.registry import METRICS


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """Confusion-matrix-based mIoU / per-class IoU.

    Args:
        num_classes: number of foreground classes.
        ignore_index: label value ignored in both gt and prediction.
        iou_metrics: which metrics to report. Only ``'mIoU'`` and per-class
            IoU are computed here; the arg is kept for parity with mmseg's
            IoUMetric config.
        prefix: prefix in the returned metric names.
        class_names: optional class names for per-class printing.
    """

    default_prefix: Optional[str] = "seg"

    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 255,
        iou_metrics: Sequence[str] = ("mIoU",),
        prefix: Optional[str] = None,
        class_names: Optional[Sequence[str]] = None,
        collect_device: str = "cpu",
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.iou_metrics = tuple(iou_metrics)
        self.class_names = list(class_names) if class_names else None

    # `process` is called per batch; we accumulate integer confusion counts.
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred_pd = sample.get("pred_sem_seg", None)
            gt_pd = sample.get("gt_sem_seg", None)
            if pred_pd is None or gt_pd is None:
                continue
            pred = _to_1d(pred_pd["sem_seg"])
            gt = _to_1d(gt_pd["sem_seg"])
            if pred.shape != gt.shape:
                gt = _resize_1d_to_pred(gt_pd["sem_seg"], pred_pd["sem_seg"])
            mask = gt != self.ignore_index
            pred = pred[mask]
            gt = gt[mask]
            # Confusion counts as a 1D histogram of (gt * K + pred).
            k = self.num_classes
            valid = (gt >= 0) & (gt < k) & (pred >= 0) & (pred < k)
            bins = np.bincount(
                gt[valid].astype(np.int64) * k + pred[valid].astype(np.int64),
                minlength=k * k,
            )
            self.results.append(bins)

    def compute_metrics(self, results: List[np.ndarray]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if not results:
            logger.warning("IoUMetric: no samples processed; returning empty.")
            return {}
        k = self.num_classes
        total = np.sum(np.stack(results, axis=0), axis=0).reshape(k, k)
        tp = np.diag(total).astype(np.float64)
        fp = total.sum(axis=0).astype(np.float64) - tp
        fn = total.sum(axis=1).astype(np.float64) - tp
        denom = tp + fp + fn
        # Guard div-by-zero: absent classes get IoU=0 and are skipped from mean.
        iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), np.nan)

        metrics: Dict[str, float] = {}
        with np.errstate(invalid="ignore"):
            metrics["mIoU"] = float(np.nanmean(iou))
        acc_denom = tp + fn
        pixel_acc = float(tp.sum() / max(1.0, total.sum()))
        metrics["aAcc"] = pixel_acc
        macc = np.where(acc_denom > 0, tp / np.maximum(acc_denom, 1e-12), np.nan)
        metrics["mAcc"] = float(np.nanmean(macc))

        for c in range(k):
            name = (self.class_names[c] if self.class_names and c < len(self.class_names)
                    else f"class_{c}")
            metrics[f"IoU.{name}"] = float(iou[c]) if not np.isnan(iou[c]) else float("nan")
        return metrics


def _to_1d(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def _resize_1d_to_pred(gt_2d, pred_2d) -> np.ndarray:
    import torch.nn.functional as F
    g = gt_2d
    if not torch.is_tensor(g):
        g = torch.from_numpy(np.asarray(g))
    p = pred_2d if torch.is_tensor(pred_2d) else torch.from_numpy(np.asarray(pred_2d))
    g = g.float()
    while g.dim() < 4:
        g = g.unsqueeze(0)
    while p.dim() < 4:
        p = p.unsqueeze(0)
    g_r = F.interpolate(g, size=p.shape[-2:], mode="nearest")
    return g_r.long().cpu().numpy().reshape(-1)
