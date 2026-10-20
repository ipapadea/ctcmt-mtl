"""Smoke test: build each source config, run one forward-backward step on
a synthetic sample, print the resulting loss dict for each task.

Run: ./docker/run.sh python docker/singletask_smoke_test.py
"""
from __future__ import annotations

import sys
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData, PixelData
from mmtrack.structures import TrackDataSample

from shift_tta.registry import MODELS
from shift_tta.utils import register_all_modules


DET_CFG = "configs/source/yolox/yolox_x_8xb4-24e_cityscapes.py"
SEG_CFG = "configs/source/yolox_seg/yolox_x_seg_8xb4-24e_cityscapes.py"


def _rand_sample(h: int, w: int, seg_classes: int) -> TrackDataSample:
    ds = TrackDataSample()
    ds.set_metainfo(dict(
        img_path="<synthetic>", img_id=0, frame_id=0, video_id=0,
        ori_shape=(h, w), img_shape=(h, w), pad_shape=(h, w),
        scale_factor=(1.0, 1.0),
    ))
    # A handful of gt boxes so YOLOX's SimOTA can assign something.
    n = 6
    xyxy = torch.zeros(n, 4)
    xyxy[:, 0] = torch.randint(0, w // 2, (n,))
    xyxy[:, 1] = torch.randint(0, h // 2, (n,))
    xyxy[:, 2] = xyxy[:, 0] + torch.randint(20, w // 4, (n,))
    xyxy[:, 3] = xyxy[:, 1] + torch.randint(20, h // 4, (n,))
    labels = torch.randint(0, 8, (n,))
    ds.gt_instances = InstanceData(bboxes=xyxy, labels=labels)
    # Ignore-region-flag list expected by mmdet
    ds.ignored_instances = InstanceData(
        bboxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long),
    )
    # Dense seg gt with a few classes + ignore region.
    mask = torch.randint(0, seg_classes, (h, w), dtype=torch.long)
    mask[: h // 8, : w // 8] = 255
    ds.gt_sem_seg = PixelData(sem_seg=mask[None])
    return ds


def _build(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    cfg.model["detector"].pop("init_cfg", None)
    if "yolox" in cfg.model["detector"] and isinstance(
            cfg.model["detector"]["yolox"], dict):
        cfg.model["detector"]["yolox"].pop("init_cfg", None)
    cfg.model["adapter"] = None
    return MODELS.build(cfg.model).cuda()


def _step(model, samples, h, w):
    B = len(samples)
    img = torch.randn(B, 3, h, w, device="cuda") * 60 + 128
    # Move any tensor fields on the sample onto CUDA so mmdet heads don't hit
    # a cross-device broadcast.
    for ds in samples:
        for name in ("gt_instances", "ignored_instances"):
            inst = getattr(ds, name, None)
            if inst is None:
                continue
            for k in list(inst.keys()):
                v = inst[k]
                if torch.is_tensor(v):
                    inst[k] = v.cuda()
        seg = getattr(ds, "gt_sem_seg", None)
        if seg is not None and torch.is_tensor(seg.sem_seg):
            seg.sem_seg = seg.sem_seg.cuda()
    losses = model.detector.loss(img, samples)
    total = sum(v for v in losses.values() if v.ndim == 0)
    total.backward()
    return {k: float(v) for k, v in losses.items() if v.ndim == 0}


def main() -> int:
    register_all_modules()
    init_default_scope("shift_tta")
    torch.manual_seed(0)
    h, w = 480, 640

    print("=== Det-only YOLOX-X (Cityscapes) ===")
    det_model = _build(DET_CFG).train()
    samples = [_rand_sample(h, w, seg_classes=19) for _ in range(2)]
    det_losses = _step(det_model, samples, h, w)
    for k, v in det_losses.items():
        print(f"  {k:30s} {v:.4f}")
    for k, v in det_losses.items():
        assert np.isfinite(v), f"non-finite det loss: {k}={v}"
    print()

    print("=== Seg-only YOLOX-X trunk + seg head (Cityscapes) ===")
    seg_model = _build(SEG_CFG).train()
    seg_losses = _step(seg_model, samples, h, w)
    for k, v in seg_losses.items():
        print(f"  {k:30s} {v:.4f}")
    for k, v in seg_losses.items():
        assert np.isfinite(v), f"non-finite seg loss: {k}={v}"

    print("\nBoth single-task configs build and backprop cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
