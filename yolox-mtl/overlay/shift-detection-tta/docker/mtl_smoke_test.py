"""End-to-end MTL smoke test:
  1. Build the full CTTA config (model + adapter + pipelines) from
     configs/continuous/ctcmt/ctcmt_yolox_mtl_cityscapes_to_acdc.py.
  2. Run a forward + adapt pass on a dummy image.
  3. Assert both `pred_det_instances` and `pred_sem_seg` end up on the sample.

Run inside the container:
    ./docker/run.sh python docker/mtl_smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmtrack.structures import TrackDataSample
from PIL import Image

from shift_tta.registry import MODELS
from shift_tta.utils import register_all_modules


CONFIG = (
    "configs/continuous/ctcmt/"
    "ctcmt_yolox_mtl_cityscapes_to_acdc.py"
)


def _make_dummy_sample(tmp_dir: str) -> TrackDataSample:
    """Write a random RGB image to disk and wrap it in a TrackDataSample with
    the metainfo a val dataloader would normally populate."""
    h, w = 320, 512
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    path = os.path.join(tmp_dir, "dummy.png")
    Image.fromarray(arr).save(path)

    ds = TrackDataSample()
    ds.set_metainfo(dict(
        img_path=path,
        img_id=0,
        frame_id=0,
        video_id=0,
        ori_shape=(h, w),
        img_shape=(h, w),
        pad_shape=(h, w),
        scale_factor=(1.0, 1.0),
    ))
    ds.instances = []
    ds.pred_det_instances = InstanceData(
        bboxes=torch.zeros((0, 4)),
        labels=torch.zeros((0,), dtype=torch.long),
        scores=torch.zeros((0,)),
    )
    return ds


def main() -> int:
    register_all_modules()
    init_default_scope("shift_tta")

    cfg = Config.fromfile(CONFIG)
    # Disable pretrained-weights download for the smoke test.
    yolox_cfg = cfg.model["detector"]["yolox"]
    yolox_cfg.pop("init_cfg", None)
    # Cheaper knobs so the smoke test fits on a single GPU.
    cfg.model["adapter"]["optim_steps"] = 1
    cfg.model["adapter"]["views"] = 1
    cfg.model["adapter"]["teacher_pipeline"][0]["scale"] = (320, 512)
    cfg.model["adapter"]["student_pipeline"][0]["scale"] = (320, 512)

    model = MODELS.build(cfg.model)
    model = model.cuda().eval()

    print(f"model type      : {type(model).__name__}")
    print(f"detector type   : {type(model.detector).__name__}")
    print(f"has seg head    : {hasattr(model.detector, 'seg_head')}")
    print(f"adapter type    : {type(model.adapter).__name__}")

    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_dummy_sample(tmp)

        # AdaptiveDetector.predict expects inputs['img'] shape (N, T, C, H, W).
        img = torch.from_numpy(np.array(Image.open(ds.img_path))).permute(2, 0, 1)
        img = img.float()[None, None].cuda()  # (1, 1, 3, H, W)

        with torch.no_grad():
            out = model.predict(inputs=dict(img=img), data_samples=[ds])

    sample = out[0]
    assert hasattr(sample, "pred_det_instances"), \
        "pred_det_instances missing on the returned sample"
    print(f"pred_det_instances: {len(sample.pred_det_instances.bboxes)} boxes")

    assert hasattr(sample, "pred_sem_seg"), \
        "pred_sem_seg missing on the returned sample (seg-head prediction not attached)"
    seg = sample.pred_sem_seg.sem_seg
    print(f"pred_sem_seg      : shape={tuple(seg.shape)} dtype={seg.dtype}")

    print("MTL end-to-end predict + adapt: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
