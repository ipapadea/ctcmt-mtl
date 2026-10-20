#!/usr/bin/env python3

import os
import json
import csv
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode


ROOT = "/workspace/output/ctta_acdc"

BASE_ROOT = os.path.join(
    ROOT, "paper26_qualdump_csc12_base_s0", "inference_by_dataset"
)

OURS_ROOT = os.path.join(
    ROOT, "paper26_qualdump_csc12_ours_l1_s0", "inference_by_dataset"
)

OUT = "/workspace/output/paper26_qualitative_selection"

DOMAINS = [
    "contrast_mtl",
    "jpeg_compression_mtl",
    "frost_mtl",
    "pixelate_mtl",
    "snow_mtl",
    "elastic_transform_mtl",
]

DET_SCORE_THR = 0.50
DET_IOU_THR = 0.50
N_CLASSES = 19
IGNORE = 255


def box_iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = aa + bb - inter

    return inter / union if union > 0 else 0.0


def gt_boxes(record):
    out = []

    for ann in record.get("annotations", []):
        bbox = BoxMode.convert(
            ann["bbox"],
            ann["bbox_mode"],
            BoxMode.XYXY_ABS,
        )

        out.append({
            "bbox": [float(x) for x in bbox],
            "category_id": int(ann["category_id"]),
        })

    return out


def det_stats(preds, gts):
    """
    Simple per-image diagnostic, NOT COCO AP.

    Predictions are filtered at score >= 0.5, then greedily matched
    class-wise at IoU >= 0.5.

    Used only for qualitative case ranking.
    """

    preds = [
        p for p in preds
        if float(p.get("score", 0.0)) >= DET_SCORE_THR
    ]

    preds = sorted(
        preds,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )

    matched_gt = set()
    tp = 0
    fp = 0

    for p in preds:
        pbox = p["bbox"]

        # COCO prediction bbox is XYWH.
        pxyxy = [
            pbox[0],
            pbox[1],
            pbox[0] + pbox[2],
            pbox[1] + pbox[3],
        ]

        best_iou = 0.0
        best_j = None

        for j, g in enumerate(gts):
            if j in matched_gt:
                continue

            if int(p["category_id"]) != int(g["category_id"]):
                continue

            iou = box_iou_xyxy(pxyxy, g["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None and best_iou >= DET_IOU_THR:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_det(path):
    rows = torch.load(path, map_location="cpu")

    return {
        int(r["image_id"]): r.get("instances", [])
        for r in rows
    }


def load_semseg_json(path):
    with open(path) as f:
        rows = json.load(f)

    grouped = defaultdict(list)

    for r in rows:
        grouped[r["file_name"]].append(r)

    return grouped


def decode_semseg(entries, height, width):
    pred = np.full((height, width), IGNORE, dtype=np.int32)

    for e in entries:
        rle = e["segmentation"]

        mask = mask_util.decode({
            "size": rle["size"],
            "counts": rle["counts"].encode("utf-8"),
        })

        pred[mask.astype(bool)] = int(e["category_id"])

    return pred


def per_image_miou(pred, gt):
    valid = gt != IGNORE

    ious = []

    for c in range(N_CLASSES):
        p = (pred == c) & valid
        g = (gt == c) & valid

        union = np.logical_or(p, g).sum()

        if union == 0:
            continue

        inter = np.logical_and(p, g).sum()

        ious.append(inter / union)

    if not ious:
        return float("nan")

    return 100.0 * float(np.mean(ious))


def main():
    os.makedirs(OUT, exist_ok=True)

    all_rows = []

    for domain in DOMAINS:
        print(f"Processing {domain} ...")

        records = DatasetCatalog.get(domain)

        by_id = {
            int(r["image_id"]): r
            for r in records
        }

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

        base_seg = load_semseg_json(
            os.path.join(
                BASE_ROOT,
                domain,
                "sem_seg_predictions.json",
            )
        )

        ours_seg = load_semseg_json(
            os.path.join(
                OURS_ROOT,
                domain,
                "sem_seg_predictions.json",
            )
        )

        for image_id, rec in by_id.items():
            file_name = rec["file_name"]
            gt_sem_path = rec["sem_seg_file_name"]

            if file_name not in base_seg or file_name not in ours_seg:
                continue

            gt = np.array(Image.open(gt_sem_path), dtype=np.int32)

            h, w = gt.shape

            bseg = decode_semseg(base_seg[file_name], h, w)
            oseg = decode_semseg(ours_seg[file_name], h, w)

            bmiou = per_image_miou(bseg, gt)
            omiou = per_image_miou(oseg, gt)

            gts = gt_boxes(rec)

            bd = det_stats(base_det.get(image_id, []), gts)
            od = det_stats(ours_det.get(image_id, []), gts)

            delta_seg = omiou - bmiou
            delta_f1 = od["f1"] - bd["f1"]
            delta_recall = od["recall"] - bd["recall"]

            # Ranking heuristic only.
            # Gives comparable weight to semantic and detection changes.
            joint_score = (
                delta_seg / 10.0
                + delta_f1
                + 0.5 * delta_recall
            )

            all_rows.append({
                "domain": domain.replace("_mtl", ""),
                "image_id": image_id,
                "file_name": file_name,
                "gt_sem_seg": gt_sem_path,

                "base_mIoU_img": bmiou,
                "ours_mIoU_img": omiou,
                "delta_mIoU_img": delta_seg,

                "base_det_f1": bd["f1"],
                "ours_det_f1": od["f1"],
                "delta_det_f1": delta_f1,

                "base_recall": bd["recall"],
                "ours_recall": od["recall"],
                "delta_recall": delta_recall,

                "base_tp": bd["tp"],
                "base_fp": bd["fp"],
                "base_fn": bd["fn"],

                "ours_tp": od["tp"],
                "ours_fp": od["fp"],
                "ours_fn": od["fn"],

                "joint_score": joint_score,
            })

    out_csv = os.path.join(OUT, "qualitative_candidates.csv")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys())
        )
        w.writeheader()
        w.writerows(all_rows)

    print()
    print("=" * 110)
    print("TOP CANDIDATES PER DOMAIN")
    print("=" * 110)

    for domain in [x.replace("_mtl", "") for x in DOMAINS]:

        rows = [
            r for r in all_rows
            if r["domain"] == domain
        ]

        if domain == "elastic_transform":
            # Failure case: most negative first.
            rows.sort(key=lambda x: x["joint_score"])
        elif domain == "snow":
            # Neutral cases: closest to zero.
            rows.sort(key=lambda x: abs(x["joint_score"]))
        else:
            rows.sort(
                key=lambda x: x["joint_score"],
                reverse=True
            )

        print()
        print(f"--- {domain} ---")

        for r in rows[:8]:
            print(
                f"id={r['image_id']:>6}  "
                f"Δseg={r['delta_mIoU_img']:+7.2f}  "
                f"ΔF1={r['delta_det_f1']:+6.3f}  "
                f"Δrec={r['delta_recall']:+6.3f}  "
                f"score={r['joint_score']:+7.3f}  "
                f"{os.path.basename(r['file_name'])}"
            )

    print()
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
