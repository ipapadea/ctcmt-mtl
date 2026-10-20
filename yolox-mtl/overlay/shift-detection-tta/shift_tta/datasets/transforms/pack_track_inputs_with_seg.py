"""Extension of ``mmtrack.PackTrackInputs`` that also packs ``gt_seg_map``.

``PackTrackInputs`` is focused on tracking targets (bboxes, labels, masks,
instance ids) and drops ``gt_seg_map`` even when it was loaded. For the
MTL/seg-only configs we need the semantic-seg gt to survive packing so the
detector's ``loss(img, data_samples)`` receives it under
``data_sample.gt_sem_seg``.
"""

from __future__ import annotations

import numpy as np
from mmcv.transforms.utils import cache_randomness  # noqa: F401  (kept for parity)
from mmengine.structures import PixelData
from mmtrack.datasets.transforms.formatting import PackTrackInputs

from shift_tta.registry import TRANSFORMS


def _to_tensor(x):
    import torch
    if torch.is_tensor(x):
        return x
    return torch.from_numpy(np.ascontiguousarray(x))


@TRANSFORMS.register_module()
class PackTrackInputsWithSeg(PackTrackInputs):
    """Like ``PackTrackInputs`` but also attaches ``gt_sem_seg`` to the
    packed ``TrackDataSample`` when ``gt_seg_map`` is present in results."""

    def transform(self, results: dict) -> dict:
        seg_map = results.get("gt_seg_map", None)
        packed = super().transform(results)
        if seg_map is None:
            return packed
        sample = packed.get("data_samples", None)
        if sample is None:
            return packed
        arr = seg_map if isinstance(seg_map, np.ndarray) else np.asarray(seg_map)
        if arr.ndim == 2:
            arr = arr[None, ...]
        sample.gt_sem_seg = PixelData(sem_seg=_to_tensor(arr.copy()))
        return packed
