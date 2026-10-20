#!/usr/bin/env python3
"""Summarize CTCMT binary adaptive resnet_route occupancy from a
retention_probe log (CTCMT_CONFLICT_MODE=resnet_route, CTCMT_ADAPTIVE_ROUTING=True).

Usage:
    python3 scripts/summarize_adaptive_routing.py LOG [--csv OUT.csv]
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

RETENTION_RE = re.compile(
    r"\[RETENTION\] kind=(\S+) state=(\d+) round=(\d+) position=(\d+) "
    r"dataset=(\S+) iter_before=(-?\d+)"
)
STEP_RE = re.compile(
    r"\[CT-CMT-MTL\] iter=(\d+) .*?"
    r"lam=([\d.]+)(?: route=resnet)?"
    r"(?: agree=([\d.]+)/([\d.]+))?"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--csv", default=None, help="optional per-step CSV dump")
    args = ap.parse_args()

    with open(args.log) as f:
        lines = f.readlines()

    # Only "adapt" segments advance model.iter; probes are frozen and do not.
    segments = []  # (iter_before, round, position, dataset)
    for line in lines:
        m = RETENTION_RE.search(line)
        if m and m.group(1) == "adapt":
            segments.append(
                (int(m.group(6)), int(m.group(3)), int(m.group(4)), m.group(5))
            )
    segments.sort(key=lambda s: s[0])

    def locate(it):
        label = None
        for seg in segments:
            if seg[0] <= it:
                label = seg
            else:
                break
        return label

    rows = []
    for line in lines:
        m = STEP_RE.search(line)
        if not m:
            continue
        it = int(m.group(1))
        lam = float(m.group(2))
        agree = float(m.group(3)) if m.group(3) else None
        peak = float(m.group(4)) if m.group(4) else None
        rows.append((it, lam, agree, peak, locate(it)))

    if not rows:
        print("No '[CT-CMT-MTL] ... lam=' lines found "
              "(wrong log, or CTCMT_CONFLICT_MODE != resnet_route).")
        sys.exit(1)

    n = len(rows)
    n_open = sum(1 for r in rows if abs(r[1] - 1.0) < 1e-6)
    n_closed = sum(1 for r in rows if abs(r[1]) < 1e-6)
    n_other = n - n_open - n_closed

    print(f"logged steps          : {n}")
    print(f"lam=1.00 (open)       : {n_open} ({100 * n_open / n:.1f}%)")
    print(f"lam=0.00 (closed)     : {n_closed} ({100 * n_closed / n:.1f}%)")
    if n_other:
        print(f"other lam values      : {n_other} ({100 * n_other / n:.1f}%)")

    with_agree = [r for r in rows if r[2] is not None]
    if with_agree:
        print()
        print("agreement trajectory (first 3 / last 3 logged steps):")
        for it, lam, agree, peak, _seg in with_agree[:3] + with_agree[-3:]:
            ratio = agree / peak if peak else float("nan")
            print(f"  iter={it:>7} lam={lam:.2f} agree={agree:.4f} "
                  f"peak={peak:.4f} ratio={ratio:.4f}")

    print()
    print("occupancy per (round, dataset):")
    buckets = defaultdict(list)
    for r in rows:
        _it, _lam, _agree, _peak, seg = r
        key = (seg[1], seg[3]) if seg else (-1, "?")
        buckets[key].append(r)
    for key in sorted(buckets):
        bucket_rows = buckets[key]
        bn = len(bucket_rows)
        bo = sum(1 for r in bucket_rows if abs(r[1] - 1.0) < 1e-6)
        bc = sum(1 for r in bucket_rows if abs(r[1]) < 1e-6)
        agrees = [r[2] for r in bucket_rows if r[2] is not None]
        mean_agree = sum(agrees) / len(agrees) if agrees else float("nan")
        print(f"  round={key[0]} dataset={key[1]:<14} n={bn:>4} "
              f"open={bo:>4} ({100 * bo / bn:.1f}%) closed={bc:>4} "
              f"({100 * bc / bn:.1f}%) mean_agree={mean_agree:.4f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "lam", "agree_ema", "agree_max", "round", "dataset"])
            for it, lam, agree, peak, seg in rows:
                w.writerow([it, lam, agree, peak,
                            seg[1] if seg else "", seg[3] if seg else ""])
        print(f"\nwrote per-step CSV: {args.csv}")


if __name__ == "__main__":
    main()
