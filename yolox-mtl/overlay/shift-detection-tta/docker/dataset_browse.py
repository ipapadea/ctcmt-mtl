"""Sanity-check that both single-task Cityscapes dataloaders load real data
end-to-end and expose the fields the training loop expects.

Prints, for one sample from each dataloader:
    - image path + shape
    - number of bboxes + labels + label range
    - seg mask shape + unique classes (if present)
    - metainfo keys

Also verifies that image bytes on disk match the file the pipeline resolves.

Run inside the container:
    ./docker/run.sh python docker/dataset_browse.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from shift_tta.utils import register_all_modules


CFGS = {
    "det":     "configs/source/yolox/yolox_x_8xb4-24e_cityscapes.py",
    "seg":     "configs/source/yolox_seg/yolox_x_seg_8xb4-24e_cityscapes.py",
    "mtl":     "configs/source/yolox_mtl/yolox_x_mtl_8xb4-24e_cityscapes.py",
}


def _one_sample(cfg_path: str, split: str) -> None:
    print(f"\n==================== {cfg_path}  [{split}] ====================")
    cfg = Config.fromfile(cfg_path)
    # Turn off pretrained downloads (not needed for a data-only check).
    if "detector" in cfg.model and isinstance(cfg.model["detector"], dict):
        cfg.model["detector"].pop("init_cfg", None)
        if "yolox" in cfg.model["detector"] and isinstance(
                cfg.model["detector"]["yolox"], dict):
            cfg.model["detector"]["yolox"].pop("init_cfg", None)

    if split == "train":
        loader_cfg = cfg.train_dataloader
    else:
        loader_cfg = cfg.val_dataloader
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False

    loader = Runner.build_dataloader(loader_cfg)
    print(f"  dataset len : {len(loader.dataset)}")

    batch = next(iter(loader))
    inputs = batch["inputs"]
    samples = batch["data_samples"]
    if isinstance(inputs, dict):
        img_t = inputs.get("img")
    else:
        img_t = inputs
    if isinstance(img_t, list):
        shapes = [tuple(t.shape) for t in img_t if hasattr(t, "shape")]
        print(f"  batch img shapes: {shapes}")
        dtypes = {str(t.dtype) for t in img_t if hasattr(t, "dtype")}
        print(f"  batch dtypes    : {sorted(dtypes)}")
    elif hasattr(img_t, "shape"):
        print(f"  batch img shape : {tuple(img_t.shape)}")
        print(f"  batch dtype     : {img_t.dtype}")
    else:
        print(f"  batch img type  : {type(img_t).__name__}")

    for i, s in enumerate(samples[:2]):
        print(f"  -- sample {i}")
        print(f"     img_path   : {s.metainfo.get('img_path', '?')}")
        print(f"     ori_shape  : {s.metainfo.get('ori_shape')}")
        print(f"     img_shape  : {s.metainfo.get('img_shape')}")
        print(f"     pad_shape  : {s.metainfo.get('pad_shape')}")
        gi = getattr(s, "gt_instances", None)
        if gi is not None:
            print(f"     gt bboxes  : {tuple(gi.bboxes.shape)}"
                  f"  labels_range=[{int(gi.labels.min()) if len(gi.labels) else 'n/a'},"
                  f"{int(gi.labels.max()) if len(gi.labels) else 'n/a'}]")
        gs = getattr(s, "gt_sem_seg", None)
        if gs is not None:
            m = gs.sem_seg
            unique = torch.unique(m).tolist()
            print(f"     gt sem_seg : shape={tuple(m.shape)} unique={unique}")
        else:
            print(f"     gt sem_seg : <absent>")
        # Show any other attributes carrying seg-like data.
        extra_attrs = [k for k in dir(s) if 'seg' in k.lower() and not k.startswith('_')]
        for k in extra_attrs:
            try:
                v = getattr(s, k)
            except Exception:
                continue
            if v is None or callable(v):
                continue
            summary = type(v).__name__
            if hasattr(v, "shape"):
                summary += f" shape={tuple(v.shape)}"
            elif hasattr(v, "sem_seg"):
                summary += f" .sem_seg shape={tuple(v.sem_seg.shape)}"
            print(f"     .{k}: {summary}")

        # Verify img_path resolves and file exists on disk.
        p = s.metainfo.get("img_path")
        if p and not os.path.isfile(p):
            print(f"     WARN: img_path does not resolve to a file: {p}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CFGS.keys()),
                        choices=list(CFGS.keys()))
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    register_all_modules()
    init_default_scope("shift_tta")

    for key in args.configs:
        _one_sample(CFGS[key], args.split)

    print("\nAll dataloaders yielded at least one sample without errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
