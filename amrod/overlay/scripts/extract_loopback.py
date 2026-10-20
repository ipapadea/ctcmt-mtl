#!/usr/bin/env python3
"""Extract mAP_Loopback and compute Forgetting = mAP_Source - mAP_Loopback.

Uses the same log-parsing engine as extract_enhancements.py but adds
support for a 5th 'domain' (cityscapes_val_mtl or cityscapes_fine_instance_seg_val)
that we treat as the loopback split.

Forgetting metric follows Moraiti et al. EJAI 2026:
    Forgetting = mAP_Source - mAP_Loopback

where mAP_Source is the source-model performance on Cityscapes val BEFORE
adaptation, and mAP_Loopback is the SAME evaluation AFTER the 4-weather
ACDC continual adaptation completes.

Negative values of Forgetting indicate the adaptation improved even the
source-domain performance (a positive sign for the framework).
"""
import os, re, statistics

HDR_RE = re.compile(r"Evaluation results for (\S+) in csv format")
TASK_RE = re.compile(r"copypaste: Task: (\S+)")
VALS_RE = re.compile(r"copypaste: ([\d.,]+)$")
SEEDS = [0, 42, 123]

# Source (pre-adaptation) numbers on Cityscapes val — measured with the
# corresponding source-only run (no CTTA loop).
SOURCE_MTL = {"AP50": 41.5, "mIoU": 69.24}       # Panoptic-FPN R-50 MTL
SOURCE_MRCNN = {"AP50": 41.2, "mIoU": None}       # Mask R-CNN R-50 FPN (det only)


def parse_log(path, loopback_name):
    if not os.path.isfile(path):
        return None
    lines = open(path).read().splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = HDR_RE.search(line)
        if not m:
            continue
        ds = m.group(1)
        for j in range(i + 1, min(i + 15, len(lines))):
            mt = TASK_RE.search(lines[j])
            if mt and j + 2 < len(lines):
                mv = VALS_RE.search(lines[j + 2])
                if not mv:
                    continue
                vals = mv.group(1).split(",")
                task = mt.group(1)
                out.setdefault(ds, {})
                if task == "sem_seg":
                    out[ds]["mIoU"] = float(vals[0])
                elif task == "bbox":
                    out[ds]["AP50"] = float(vals[1])
    return out.get(loopback_name)


VARIANTS = [
    ("CT-CMT-MTL + V2 + CTPV (headline)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_loopback/mtl_v2_ctpv_seed{s}.log",
     "cityscapes_val_mtl",
     SOURCE_MTL),
    ("CTCMT_Det on MR-CNN",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_loopback/det_mr_seed{s}.log",
     "cityscapes_fine_instance_seg_val",
     SOURCE_MRCNN),
]


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def render():
    print("\n### Loopback (mAP_Source vs mAP_Loopback on Cityscapes val)")
    print()
    print("| Method | Source AP50 | Loopback AP50 | Forgetting AP50 | Source mIoU | Loopback mIoU | Forgetting mIoU |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, pattern, ds_name, src in VARIANTS:
        aps, mious = [], []
        for s in SEEDS:
            r = parse_log(pattern.format(s=s), ds_name)
            if r is None:
                continue
            if "AP50" in r:
                aps.append(r["AP50"])
            if "mIoU" in r:
                mious.append(r["mIoU"])
        loopback_ap = fmt(aps)
        loopback_mi = fmt(mious)
        src_ap = f"{src['AP50']:.2f}" if src["AP50"] is not None else "—"
        src_mi = f"{src['mIoU']:.2f}" if src["mIoU"] is not None else "—"
        forget_ap = "—"
        forget_mi = "—"
        if aps:
            deltas = [src["AP50"] - v for v in aps if v is not None]
            if len(deltas) > 1:
                forget_ap = f"{statistics.mean(deltas):+.2f} ± {statistics.stdev(deltas):.2f}"
            elif deltas:
                forget_ap = f"{deltas[0]:+.2f}"
        if mious and src["mIoU"]:
            deltas = [src["mIoU"] - v for v in mious if v is not None]
            if len(deltas) > 1:
                forget_mi = f"{statistics.mean(deltas):+.2f} ± {statistics.stdev(deltas):.2f}"
            elif deltas:
                forget_mi = f"{deltas[0]:+.2f}"
        print(f"| {name} | {src_ap} | {loopback_ap} | {forget_ap} | {src_mi} | {loopback_mi} | {forget_mi} |")

    print("""
Interpretation:
  - Forgetting = Source - Loopback. Positive => the model has FORGOTTEN.
  - Negative => the model performs BETTER on the source domain after
    the 4-weather adaptation than before. Consistent with the prior
    EJAI 2026 finding that MT+SR+CL yields negative Forgetting on KITTI.
  - The loopback evaluation is with adaptation still ON (the model
    processes Cityscapes val images as if they were a 5th weather).
    This matches the "loopback" convention in Moraiti et al. 2026
    where mAP_Loopback is computed after the full adaptation sequence.
""")


if __name__ == "__main__":
    render()
