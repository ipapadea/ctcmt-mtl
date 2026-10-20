#!/usr/bin/env python3
"""Extract detectron2 'copypaste' metric blocks from run logs.

Handles both shapes:
  - a source-model eval  (one or more datasets, evaluated once each)
  - a CTTA run           (one dataset per domain visit, 40/50 of them)

Usage:
  python3 scripts/gather_eval.py LOG [LOG ...]            # per-evaluation rows
  python3 scripts/gather_eval.py --mean LOG [LOG ...]     # mean over evaluations
"""
import re
import sys
from collections import OrderedDict

CSV_RE = re.compile(r"Evaluation results for (\S+) in csv format:")
TASK_RE = re.compile(r"copypaste: Task: (\S+)")
HEAD_RE = re.compile(r"copypaste: ([A-Za-z].*)$")
VALS_RE = re.compile(r"copypaste: ([-\d.]+(?:,[-\d.]+)*)$")

WANT = [("sem_seg", "mIoU"), ("bbox", "AP"), ("bbox", "AP50")]


def parse(path):
    """-> list of (dataset, {(task, metric): value})."""
    out, cur, task, head = [], None, None, None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = CSV_RE.search(line)
        if m:
            cur = (m.group(1), {})
            out.append(cur)
            task = head = None
            continue
        if cur is None:
            continue
        m = TASK_RE.search(line)
        if m:
            task, head = m.group(1), None
            continue
        m = VALS_RE.search(line)
        if m and head:
            for k, v in zip(head, m.group(1).split(",")):
                try:
                    cur[1][(task, k)] = float(v)
                except ValueError:
                    pass
            head = None
            continue
        m = HEAD_RE.search(line)
        if m and task:
            head = [h.strip() for h in m.group(1).split(",")]
    return out


def main(argv):
    mean = "--mean" in argv
    logs = [a for a in argv if not a.startswith("--")]
    cols = [f"{t}.{k}" for t, k in WANT]
    print(f"{'log':<38} {'dataset/n':<26} " + " ".join(f"{c:>12}" for c in cols))
    print("-" * (38 + 26 + 13 * len(cols)))
    for path in logs:
        evals = parse(path)
        name = path.split("/")[-1].replace(".log", "")
        if not evals:
            print(f"{name:<38} {'NO RESULTS':<26}")
            continue
        if mean:
            agg = OrderedDict()
            for _, d in evals:
                for key in WANT:
                    if key in d:
                        agg.setdefault(key, []).append(d[key])
            cells = []
            for key in WANT:
                v = agg.get(key)
                cells.append(f"{sum(v)/len(v):12.3f}" if v else " " * 12)
            print(f"{name:<38} {'mean of %d' % len(evals):<26} " + " ".join(cells))
        else:
            for dset, d in evals:
                cells = [
                    f"{d[key]:12.3f}" if key in d else " " * 12 for key in WANT
                ]
                print(f"{name:<38} {dset:<26} " + " ".join(cells))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1:])
