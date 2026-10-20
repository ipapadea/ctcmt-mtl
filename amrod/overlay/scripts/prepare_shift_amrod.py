#!/usr/bin/env python3
"""Prepare SHIFT annotation files for AMROD detectron2 CTTA experiments.

Reads the SHIFT COCO-format detection JSONs and writes per-condition splits
that match the path structure expected by register_shift() in builtin.py:

  /data/ilias/shift_amrod/
    annotations/
      gtfine_clear_train.json      <- discrete train, clear only (source domain)
      gtfine_cloudy_val.json       <- discrete val, cloudy
      gtfine_overcast_val.json     <- discrete val, overcast
      gtfine_rainy_val.json        <- discrete val, rainy
      gtfine_foggy_val.json        <- discrete val, foggy
    rgb_anon/images/
      train/front -> symlink to discrete train img/
      val/front   -> symlink to discrete val img/

Run once: python3 prepare_shift_amrod.py
"""

import json
import os
import sys
from pathlib import Path

SHIFT_ROOT = Path("/data/ilias/shift")
OUT_ROOT   = Path("/data/ilias/shift_amrod")

DISCRETE_TRAIN_JSON  = SHIFT_ROOT / "discrete/images/train/front/det_2d_cocoformat.json"
DISCRETE_TRAIN_IMGS  = SHIFT_ROOT / "discrete/images/train/front/img"
DISCRETE_VAL_JSON    = SHIFT_ROOT / "discrete/images/val/front/det_2d_cocoformat.json"
DISCRETE_VAL_IMGS    = SHIFT_ROOT / "discrete/images/val/front/img"


def filter_coco(coco: dict, weather: str) -> dict:
    """Return a new COCO dict keeping only images with weather_coarse == weather."""
    keep_imgs = [img for img in coco["images"]
                 if img["attributes"]["weather_coarse"] == weather]
    keep_ids  = {img["id"] for img in keep_imgs}
    keep_anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    return {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "categories":  coco["categories"],
        "images":      keep_imgs,
        "annotations": keep_anns,
    }


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    n_imgs = len(obj["images"])
    n_anns = len(obj["annotations"])
    print(f"  wrote {path.name}  ({n_imgs} images, {n_anns} annotations)")


def make_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    print(f"  symlink {dst} -> {src}")


def main():
    print("Loading discrete train JSON …")
    train = json.load(open(DISCRETE_TRAIN_JSON))
    print(f"  {len(train['images'])} images total")

    print("Loading discrete val JSON …")
    val = json.load(open(DISCRETE_VAL_JSON))
    print(f"  {len(val['images'])} images total")

    ann_dir = OUT_ROOT / "annotations"

    # Source domain: clear training images
    print("\nWriting train / val splits:")
    write_json(filter_coco(train, "clear"),    ann_dir / "gtfine_clear_train.json")
    write_json(filter_coco(val,   "cloudy"),   ann_dir / "gtfine_cloudy_val.json")
    write_json(filter_coco(val,   "overcast"), ann_dir / "gtfine_overcast_val.json")
    write_json(filter_coco(val,   "rainy"),    ann_dir / "gtfine_rainy_val.json")
    write_json(filter_coco(val,   "foggy"),    ann_dir / "gtfine_foggy_val.json")

    print("\nCreating image symlinks:")
    make_symlink(DISCRETE_TRAIN_IMGS, OUT_ROOT / "rgb_anon/images/train/front")
    make_symlink(DISCRETE_VAL_IMGS,   OUT_ROOT / "rgb_anon/images/val/front")

    print("\nDone. Dataset root: /data/ilias/shift_amrod")
    print("Add this docker volume mount to run_ctta_acdc.sh:")
    print("  -v /data/ilias/shift_amrod:/datasets/shift:ro \\")


if __name__ == "__main__":
    main()
