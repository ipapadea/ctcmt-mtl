#!/usr/bin/env python3

import os
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


ROOT = "/workspace/output/ctta_acdc"

BASE_ROOT = os.path.join(
    ROOT, "paper26_qualdump_csc12_base_s0", "inference_by_dataset"
)

OURS_ROOT = os.path.join(
    ROOT, "paper26_qualdump_csc12_ours_l1_s0", "inference_by_dataset"
)

OUT = "/workspace/output/paper26_qualitative_selection/rendered"

SCORE_THR = 0.50
ALPHA = 0.45


CASES = [
    ("contrast_mtl",            315, "strong_joint"),
    ("jpeg_compression_mtl",    322, "joint"),
    ("frost_mtl",               332, "hard_corruption"),
    ("pixelate_mtl",            499, "seg_heavy"),
    ("snow_mtl",                443, "neutral"),
    ("elastic_transform_mtl",   306, "failure"),
]


# Cityscapes trainId palette
SEM_COLORS = np.array([
    [128,  64, 128],   # road
    [244,  35, 232],   # sidewalk
    [ 70,  70,  70],   # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170,  30],   # traffic light
    [220, 220,   0],   # traffic sign
    [107, 142,  35],   # vegetation
    [152, 251, 152],   # terrain
    [ 70, 130, 180],   # sky
    [220,  20,  60],   # person
    [255,   0,   0],   # rider
    [  0,   0, 142],   # car
    [  0,   0,  70],   # truck
    [  0,  60, 100],   # bus
    [  0,  80, 100],   # train
    [  0,   0, 230],   # motorcycle
    [119,  11,  32],   # bicycle
], dtype=np.uint8)


def load_det(path):
    rows = torch.load(path, map_location="cpu")
    return {
        int(r["image_id"]): r.get("instances", [])
        for r in rows
    }


def load_sem(path):
    with open(path) as f:
        rows = json.load(f)

    by_file = defaultdict(list)

    for r in rows:
        by_file[r["file_name"]].append(r)

    return by_file


def decode_sem(entries, h, w):
    pred = np.full((h, w), 255, dtype=np.uint8)

    for e in entries:
        rle = e["segmentation"]

        mask = mask_util.decode({
            "size": rle["size"],
            "counts": rle["counts"].encode("utf-8"),
        })

        pred[mask.astype(bool)] = int(e["category_id"])

    return pred


def overlay_sem(image, mask):
    out = image.copy()

    valid = (mask >= 0) & (mask < 19)

    color = np.zeros_like(image)

    for c in range(19):
        color[mask == c] = SEM_COLORS[c][::-1]  # RGB -> BGR

    out[valid] = (
        (1.0 - ALPHA) * out[valid].astype(np.float32)
        + ALPHA * color[valid].astype(np.float32)
    ).astype(np.uint8)

    return out


def draw_pred_boxes(image, predictions, class_names):
    out = image.copy()

    for p in predictions:
        score = float(p.get("score", 0.0))

        if score < SCORE_THR:
            continue

        x, y, w, h = p["bbox"]

        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        cid = int(p["category_id"])

        label = (
            class_names[cid]
            if 0 <= cid < len(class_names)
            else str(cid)
        )

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            out,
            f"{label} {score:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_gt_boxes(image, annotations, class_names):
    out = image.copy()

    for ann in annotations:
        box = BoxMode.convert(
            ann["bbox"],
            ann["bbox_mode"],
            BoxMode.XYXY_ABS,
        )

        x1, y1, x2, y2 = [int(round(v)) for v in box]

        cid = int(ann["category_id"])

        label = (
            class_names[cid]
            if 0 <= cid < len(class_names)
            else str(cid)
        )

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def add_title(image, title):
    h, w = image.shape[:2]

    header = np.zeros((45, w, 3), dtype=np.uint8)
    header[:] = 255

    cv2.putText(
        header,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([header, image])


def main():
    os.makedirs(OUT, exist_ok=True)

    for domain, image_id, role in CASES:

        records = DatasetCatalog.get(domain)
        metadata = MetadataCatalog.get(domain)

        rec = next(
            r for r in records
            if int(r["image_id"]) == image_id
        )

        thing_classes = list(metadata.thing_classes)

        image = cv2.imread(rec["file_name"])

        if image is None:
            raise FileNotFoundError(rec["file_name"])

        h, w = image.shape[:2]

        gt_sem = np.array(
            Image.open(rec["sem_seg_file_name"]),
            dtype=np.uint8,
        )

        base_sem_json = load_sem(
            os.path.join(
                BASE_ROOT,
                domain,
                "sem_seg_predictions.json",
            )
        )

        ours_sem_json = load_sem(
            os.path.join(
                OURS_ROOT,
                domain,
                "sem_seg_predictions.json",
            )
        )

        base_det = load_det(
            os.path.join(
                BASE_ROOT,
                domain,
                "instances_predictions.pth",
            )
        )

        ours_det = load_det(
            os.path.join(
                OURS_ROOT,
                domain,
                "instances_predictions.pth",
            )
        )

        file_name = rec["file_name"]

        base_mask = decode_sem(
            base_sem_json[file_name],
            h,
            w,
        )

        ours_mask = decode_sem(
            ours_sem_json[file_name],
            h,
            w,
        )

        # Input
        panel_input = image.copy()

        # Ground truth
        panel_gt = overlay_sem(image, gt_sem)
        panel_gt = draw_gt_boxes(
            panel_gt,
            rec.get("annotations", []),
            thing_classes,
        )

        # Base
        panel_base = overlay_sem(image, base_mask)
        panel_base = draw_pred_boxes(
            panel_base,
            base_det.get(image_id, []),
            thing_classes,
        )

        # Ours
        panel_ours = overlay_sem(image, ours_mask)
        panel_ours = draw_pred_boxes(
            panel_ours,
            ours_det.get(image_id, []),
            thing_classes,
        )

        panels = [
            add_title(panel_input, "Corrupted input"),
            add_title(panel_gt, "Ground truth"),
            add_title(panel_base, "Base"),
            add_title(panel_ours, "Ours"),
        ]

        composite = np.hstack(panels)

        short_domain = domain.replace("_mtl", "")

        out = os.path.join(
            OUT,
            f"{short_domain}_id{image_id}_{role}.jpg",
        )

        cv2.imwrite(
            out,
            composite,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        print(out)


if __name__ == "__main__":
    main()
