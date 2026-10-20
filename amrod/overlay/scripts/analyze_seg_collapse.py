#!/usr/bin/env python3
"""Diagnose the semantic-segmentation collapse in the Cityscapes-C long-term stream.

Parses per-evaluation per-class IoU / BoundaryIoU already present in the logs
(no GPU needed) and tests three competing explanations:

  1. class collapse   - rare classes decay to ~0 while frequent ones hold
  2. uniform decay    - every class fades together (confidence/entropy collapse)
  3. boundary erosion - IoU holds but BoundaryIoU falls (over-smoothed teacher)

Stream layout: (fog, motion_blur, snow, brightness, defocus_blur) x 10, so
evaluation i is round i//5, domain i%5. Rounds are compared within the same
domain so domain difficulty never confounds the trend.

Usage:
  python scripts/analyze_seg_collapse.py LOG [LOG ...] [--source SOURCE_LOG]
"""
import argparse
import os
import re

DOMAINS = ["fog", "motion_blur", "snow", "brightness", "defocus_blur"]

# Cityscapes trainIds, split by pixel frequency for the class-collapse test.
FREQUENT = {"road", "building", "vegetation", "sky", "car", "sidewalk"}

IOU_RE = re.compile(r"'IoU-([^']+)': ([-\d.eE]+|nan)")
BIOU_RE = re.compile(r"'BoundaryIoU-([^']+)': ([-\d.eE]+|nan)")
MIOU_RE = re.compile(r"'mIoU': ([-\d.eE]+|nan)")


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except ValueError:
        return None


def parse(path):
    """Return list of per-evaluation dicts: {mIoU, iou{}, biou{}}."""
    evals = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "sem_seg_evaluation" not in line or "'mIoU'" not in line:
                continue
            m = MIOU_RE.search(line)
            if not m:
                continue
            evals.append({
                "mIoU": _f(m.group(1)),
                "iou": {k: _f(v) for k, v in IOU_RE.findall(line)},
                "biou": {k: _f(v) for k, v in BIOU_RE.findall(line)},
            })
    return evals


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def round_means(evals, key="mIoU"):
    """Mean of `key` per round (5 domains per round)."""
    out = []
    for r in range(len(evals) // len(DOMAINS)):
        chunk = evals[r * len(DOMAINS):(r + 1) * len(DOMAINS)]
        out.append(mean([e[key] for e in chunk]))
    return out


def class_round(evals, cls, rnd, field="iou"):
    chunk = evals[rnd * len(DOMAINS):(rnd + 1) * len(DOMAINS)]
    return mean([e[field].get(cls) for e in chunk])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--source", help="source-only log for the no-adaptation reference")
    args = ap.parse_args()

    src = {}
    if args.source and os.path.isfile(args.source):
        # Source-only log evaluates the 12 corruptions once, in config order.
        order = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow",
                 "frost", "fog", "brightness", "contrast", "elastic_transform",
                 "pixelate", "jpeg_compression"]
        se = parse(args.source)
        if len(se) == len(order):
            by_dom = dict(zip(order, se))
            classes = sorted(se[0]["iou"])
            src = {c: mean([by_dom[d]["iou"].get(c) for d in DOMAINS]) for c in classes}
            src["__mIoU__"] = mean([by_dom[d]["mIoU"] for d in DOMAINS])

    for path in args.logs:
        evals = parse(path)
        n_rounds = len(evals) // len(DOMAINS)
        if n_rounds < 2:
            print(f"\n{os.path.basename(path)}: only {len(evals)} evals, skipping")
            continue

        print("\n" + "=" * 78)
        print(f"{os.path.basename(path)}   ({len(evals)} evals = {n_rounds} rounds)")
        print("=" * 78)

        rm = round_means(evals)
        print("mIoU per round: " + " ".join(f"{v:5.2f}" for v in rm))
        if "__mIoU__" in src:
            print(f"source-only (same 5 corruptions): {src['__mIoU__']:.2f}"
                  f"   -> R{n_rounds} is {rm[-1] - src['__mIoU__']:+.2f} vs source")

        classes = sorted(evals[0]["iou"])
        rows = []
        for c in classes:
            r1 = class_round(evals, c, 0)
            rN = class_round(evals, c, n_rounds - 1)
            rows.append((c, r1, rN, rN - r1, src.get(c)))
        rows.sort(key=lambda r: r[3])

        print(f"\n{'class':<16}{'R1':>7}{'R10':>7}{'delta':>8}{'source':>8}{'R10-src':>9}")
        for c, r1, rN, d, s in rows:
            ss = f"{s:8.2f}" if s is not None else f"{'-':>8}"
            ds = f"{rN - s:+9.2f}" if s is not None else f"{'-':>9}"
            print(f"{c:<16}{r1:7.2f}{rN:7.2f}{d:+8.2f}{ss}{ds}")

        # --- hypothesis tests -------------------------------------------------
        freq = [r for r in rows if r[0] in FREQUENT]
        rare = [r for r in rows if r[0] not in FREQUENT]
        d_freq = mean([r[3] for r in freq])
        d_rare = mean([r[3] for r in rare])
        dead = [r[0] for r in rows if r[2] is not None and r[2] < 1.0]
        dead_r1 = [r[0] for r in rows if r[1] is not None and r[1] < 1.0]

        b1 = mean([class_round(evals, c, 0, "biou") for c in classes])
        bN = mean([class_round(evals, c, n_rounds - 1, "biou") for c in classes])

        print(f"\n  frequent classes  mean delta = {d_freq:+.2f}  (n={len(freq)})")
        print(f"  rare classes      mean delta = {d_rare:+.2f}  (n={len(rare)})")
        print(f"  classes < 1.0 IoU : R1 {len(dead_r1)} -> R{n_rounds} {len(dead)}"
              f"   newly dead: {sorted(set(dead) - set(dead_r1))}")
        print(f"  mean BoundaryIoU  : R1 {b1:.2f} -> R{n_rounds} {bN:.2f} ({bN - b1:+.2f})")
        print(f"  mean IoU          : R1 {mean([r[1] for r in rows]):.2f} -> "
              f"R{n_rounds} {mean([r[2] for r in rows]):.2f}")

        verdict = []
        if d_rare < d_freq - 2.0:
            verdict.append("CLASS COLLAPSE (rare classes decay much faster)")
        if abs(d_rare - d_freq) <= 2.0 and d_freq < -1.0:
            verdict.append("UNIFORM DECAY (all classes fade together)")
        if (bN - b1) < (mean([r[2] for r in rows]) - mean([r[1] for r in rows])) - 2.0:
            verdict.append("BOUNDARY EROSION (BoundaryIoU falls faster than IoU)")
        print("  => " + ("; ".join(verdict) if verdict else "no clear single pattern"))


if __name__ == "__main__":
    main()
