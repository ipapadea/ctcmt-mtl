#!/usr/bin/env python3
"""Diagnose the Cityscapes-C long-term detection plateau.

Our AP50 leads AMROD through round 4 and then flattens at 25-26 while AMROD
keeps climbing to 30.8. Five of our variants hit the same ceiling regardless of
augmentation, class balancing, CT-CR mode or restoration rule, which suggests
something structural in the detection branch rather than a hyperparameter.

Everything here comes from the periodic progress lines already in the logs.

  CTCMT_MTL prints unconditionally every 50 iters:
      [CT-CMT-MTL] iter=N score_em=X n_pseudo=Y det/loss_...=.. seg/soft_ce=..
    -> absence of any det/ key means detection contributed nothing that step
       (score-EMA gate closed, or zero surviving pseudo-boxes)

  AMROD prints its loss line only AFTER the early returns, so the number of
  loss lines per round measures how often it actually took a step.

Reported per round (2500 iters = 5 domains x 500 images):
  det_rate   fraction of sampled steps where the detection loss was present
  n_pseudo   mean surviving pseudo-boxes per image
  score_em   mean teacher-confidence EMA (drives the gate)
  thresholds mean dynamic per-class threshold (AMROD only)

Usage:
  python3 scripts/diagnose_plateau.py LOG [LOG ...] [--iters-per-round 2500]
"""
import argparse
import os
import re
from collections import defaultdict

CTCMT_RE = re.compile(r"\[CT-CMT-MTL\] iter=(\d+) score_em=([\d.]+) n_pseudo=(\d+)(.*)")
THR_RE = re.compile(r"thr=([\d.]+)/([\d.]+)/([\d.]+)")
AMROD_SCORE_RE = re.compile(r"iter:\s+(\d+) score_em ([\d.]+)")
AMROD_LOSS_RE = re.compile(r"iter:\s+(\d+) (loss_\w+:|AMROD:|st_const:)")
AMROD_THR_RE = re.compile(r"iter:\s+(\d+) thresholds:\s+\[([^\]]+)\]")


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def parse(path, ipr):
    per_round = defaultdict(lambda: {"det": [], "npseudo": [], "score": [],
                                     "thr": [], "loss_lines": 0, "score_lines": 0})
    kind = None
    for line in open(path, errors="ignore"):
        m = CTCMT_RE.search(line)
        if m:
            kind = "ctcmt"
            it, score, npse, rest = int(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4)
            r = min(it // ipr, 9)
            d = per_round[r]
            d["det"].append(1.0 if "det/" in rest else 0.0)
            d["npseudo"].append(npse)
            d["score"].append(score)
            mt = THR_RE.search(rest)
            if mt:
                d["thr"].append(float(mt.group(2)))
            continue
        m = AMROD_SCORE_RE.search(line)
        if m:
            kind = kind or "amrod"
            it, score = int(m.group(1)), float(m.group(2))
            d = per_round[min(it // ipr, 9)]
            d["score"].append(score)
            d["score_lines"] += 1
            continue
        m = AMROD_LOSS_RE.search(line)
        if m:
            per_round[min(int(m.group(1)) // ipr, 9)]["loss_lines"] += 1
            continue
        m = AMROD_THR_RE.search(line)
        if m:
            vals = [float(x) for x in m.group(2).replace(",", " ").split()]
            if vals:
                per_round[min(int(m.group(1)) // ipr, 9)]["thr"].append(sum(vals) / len(vals))
    return kind, per_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--iters-per-round", type=int, default=2500)
    args = ap.parse_args()

    for path in args.logs:
        kind, pr = parse(path, args.iters_per_round)
        if not pr:
            print(f"\n{os.path.basename(path)}: no progress lines found")
            continue
        print("\n" + "=" * 78)
        print(f"{os.path.basename(path)}   [{kind}]")
        print("=" * 78)
        rounds = sorted(pr)
        if kind == "ctcmt":
            print(f"{'round':>6}{'det_rate':>10}{'n_pseudo':>10}{'score_em':>10}{'mean_thr':>10}{'samples':>9}")
            for r in rounds:
                d = pr[r]
                thr = mean(d['thr'])
                print(f"{r+1:>6}{mean(d['det']) or 0:>10.3f}{mean(d['npseudo']) or 0:>10.2f}"
                      f"{mean(d['score']) or 0:>10.3f}"
                      f"{(f'{thr:.3f}' if thr is not None else '-'):>10}{len(d['det']):>9}")
            first, last = pr[rounds[0]], pr[rounds[-1]]
            print(f"\n  det_rate  R1 {mean(first['det']):.3f} -> R{rounds[-1]+1} {mean(last['det']):.3f}"
                  f"   ({(mean(last['det']) - mean(first['det'])):+.3f})")
            print(f"  n_pseudo  R1 {mean(first['npseudo']):.2f} -> R{rounds[-1]+1} {mean(last['npseudo']):.2f}"
                  f"   ({(mean(last['npseudo']) - mean(first['npseudo'])):+.2f})")
            print(f"  score_em  R1 {mean(first['score']):.3f} -> R{rounds[-1]+1} {mean(last['score']):.3f}")
        else:
            print(f"{'round':>6}{'step_rate':>11}{'score_em':>10}{'mean_thr':>10}")
            for r in rounds:
                d = pr[r]
                poss = args.iters_per_round / 50.0
                print(f"{r+1:>6}{d['loss_lines']/poss:>11.3f}{mean(d['score']) or 0:>10.3f}"
                      f"{mean(d['thr']) or 0:>10.3f}")


if __name__ == "__main__":
    main()
