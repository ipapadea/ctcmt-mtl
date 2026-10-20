#!/usr/bin/env python3
"""Analyse the CT-CR prior diagnostics produced by ctcmt_diag_ctcr_priors.yaml.

Two independent questions, from two sources:

1. Gradient share (from the run log's [CT-CMT-GRADDIAG] lines). How large is each
   auxiliary objective's shared-trunk gradient relative to detection's? Decides
   whether reweighting CT-CR can matter at all.

2. Spatial support (from the CTCMT_DIAG_JSONL "masked_ctcr" records). Which
   candidate CT-CR spatial prior best covers the semantic GT inside a pseudo-box:
     rect   - the whole rectangle            (mode A/A2, detector geometry)
     sem    - teacher semantic posterior     (modes B/C/D, segmentation-derived)
     mask   - teacher mask head              (detector-derived, unused today)

Usage:
  python3 scripts/analyze_ctcr_priors.py LOG JSONL
"""
import json
import re
import sys
from collections import defaultdict

GRAD_RE = re.compile(r"\[CT-CMT-GRADDIAG\] iter=(\d+) (.*)")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=(-?\d+\.?\d*)")


def _fmt(x, w=8, p=3):
    return f"{x:{w}.{p}f}" if x is not None else " " * w


def grad_share(log_path):
    vals = defaultdict(list)
    for line in open(log_path, encoding="utf-8", errors="ignore"):
        m = GRAD_RE.search(line)
        if not m:
            continue
        for k, v in KV_RE.findall(m.group(2)):
            vals[k].append(float(v))
    if not vals:
        print("No [CT-CMT-GRADDIAG] lines found.\n")
        return

    n = len(vals.get("gnorm_det", []))
    print(f"## 1. Shared-trunk gradient share   (n={n} sampled steps)\n")
    det = vals.get("gnorm_det", [])
    mean_det = sum(det) / max(len(det), 1)
    print(f"{'component':<10} {'mean |g|':>10} {'median':>10} {'vs det':>8} {'share':>8}")
    norms = {k[len("gnorm_"):]: v for k, v in vals.items() if k.startswith("gnorm_")}
    total_aux = sum(
        sum(v) / len(v) for k, v in norms.items() if k != "det"
    )
    for k in sorted(norms, key=lambda k: -sum(norms[k]) / len(norms[k])):
        v = sorted(norms[k])
        mean = sum(v) / len(v)
        med = v[len(v) // 2]
        share = "" if k == "det" else f"{mean / max(total_aux, 1e-9):7.1%}"
        print(f"{k:<10} {mean:10.3f} {med:10.3f} {mean / max(mean_det, 1e-9):7.2f}x {share:>8}")
    print(f"\n  aux total / det = {total_aux / max(mean_det, 1e-9):.2f}x")

    print(f"\n{'cos(det, .)':<14} {'mean':>8} {'median':>8} {'frac<0':>8}")
    for k in sorted(vals):
        if not k.startswith("cos_det_"):
            continue
        v = vals[k]
        neg = sum(1 for x in v if x < 0) / len(v)
        print(f"{k[len('cos_det_'):]:<14} {sum(v)/len(v):8.4f} "
              f"{sorted(v)[len(v)//2]:8.4f} {neg:7.1%}")
    blocks = {k: v for k, v in vals.items() if k.startswith("blk_")}
    if blocks:
        print(f"\n{'block':<10} {'mean cos':>10} {'frac<0':>8}   (where the conflict lives)")
        for k in sorted(blocks):
            v = blocks[k]
            print(f"{k[len('blk_'):]:<10} {sum(v)/len(v):10.4f} "
                  f"{sum(1 for x in v if x < 0)/len(v):7.1%}")
    print()


def _domain(file_name):
    parts = str(file_name or "").split("/")
    return parts[2] if len(parts) > 3 and parts[1] == "datasets" else "?"


def spatial_priors(jsonl_path):
    # acc[(domain, prior, tau)] = [inter, pred_px, gt_px]
    acc = defaultdict(lambda: [0, 0, 0])
    by_score = defaultdict(lambda: [0, 0, 0])
    n_box = 0
    taus = set()

    for line in open(jsonl_path, encoding="utf-8", errors="ignore"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "masked_ctcr":
            continue
        dom = _domain(rec.get("file_name"))
        for b in rec.get("boxes", []):
            n_box += 1
            box_px, gt_px = b["box_pixels"], b["gt_pixels"]
            # Rectangle covers the whole box, so it captures every GT pixel in it.
            for d in (dom, "ALL"):
                a = acc[(d, "rect", None)]
                a[0] += gt_px
                a[1] += box_px
                a[2] += gt_px
            for prior, key in (("sem", "sweep"), ("mask", "mask_sweep")):
                for s in b.get(key) or []:
                    taus.add(s["tau"])
                    for d in (dom, "ALL"):
                        a = acc[(d, prior, s["tau"])]
                        a[0] += s["intersection"]
                        a[1] += s["mask_pixels"]
                        a[2] += gt_px
            # Does a more confident detection give a purer box? (tests D2)
            bucket = min(int(b["det_score"] * 10) / 10, 0.9)
            a = by_score[bucket]
            a[0] += gt_px
            a[1] += box_px
            a[2] += gt_px

    if not n_box:
        print("No masked_ctcr records found.\n")
        return

    print(f"## 2. CT-CR spatial support vs semantic GT   ({n_box} pseudo-boxes)\n")
    print("   precision = GT-class pixels / supervised pixels  (purity of the supervision)")
    print("   recall    = supervised GT pixels / GT pixels in box")
    print("   IoU       = agreement of the supervised region with the GT region\n")

    doms = sorted({d for (d, _, _) in acc} - {"ALL"}) + ["ALL"]
    for d in doms:
        print(f"### {d}")
        print(f"{'prior':<8} {'tau':>5} {'precision':>10} {'recall':>8} {'IoU':>8} {'px/box':>9}")
        rows = [("rect", None)] + [
            (p, t) for p in ("sem", "mask") for t in sorted(taus)
        ]
        for prior, tau in rows:
            inter, pred, gt = acc.get((d, prior, tau), [0, 0, 0])
            if pred == 0 and gt == 0:
                continue
            prec = inter / pred if pred else 0.0
            rec = inter / gt if gt else 0.0
            iou = inter / (pred + gt - inter) if (pred + gt - inter) else 0.0
            nb = sum(1 for k in acc if k[0] == d)
            print(f"{prior:<8} {'-' if tau is None else f'{tau:.2f}':>5} "
                  f"{prec:10.4f} {rec:8.4f} {iou:8.4f} {pred / max(n_box, 1):9.1f}")
        print()

    print("### Box purity vs detector confidence   (is a more confident box a purer one?)")
    print(f"{'det_score':>10} {'precision':>10} {'n_px':>12}")
    for b in sorted(by_score):
        inter, pred, _ = by_score[b]
        print(f"{b:10.1f} {inter / max(pred, 1):10.4f} {pred:12d}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    grad_share(sys.argv[1])
    spatial_priors(sys.argv[2])
