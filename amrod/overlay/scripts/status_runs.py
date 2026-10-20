#!/usr/bin/env python3
"""Inventory of CTTA runs from their logs.

Distinguishes the three failure modes that look alike:
  OK          expected number of evaluations present
  FAILED      a Python traceback in the log
  INCOMPLETE  stopped early with no traceback (killed / SIGHUP / closed tmux)
  MISSING     no log at all

Usage:
  python3 scripts/status_runs.sh.py --logs DIR [--expect NAME=N ...]
  python3 scripts/status_runs.sh.py --logs DIR --manifest remote
"""
import argparse
import glob
import os
import re

# protocol -> number of evaluations a complete run must produce
PROTOCOL_EVALS = [
    ("acdc_lt", 40),   # 4 conditions x 10 rounds
    ("csc_lt", 50),    # 5 corruptions x 10 rounds
    ("cscLT", 50),
    ("csc12", 12),     # 12 corruptions once
    ("cs_c", 12),
    ("acdc", 4),       # 4 conditions once
]

MANIFESTS = {
    # same-source baselines that should exist on the remote machine
    "remote": [
        "amrod_pfnsrc_acdc_lt_s0", "cotta_pfnsrc_acdc_lt_s0", "tent_pfnsrc_acdc_lt_s0",
        "amrod_pfnsrc_csc_lt_s0", "cotta_pfnsrc_csc_lt_s0", "tent_pfnsrc_csc_lt_s0",
        "amrod_pfnsrc_csc12_s0", "cotta_pfnsrc_csc12_s0", "tent_pfnsrc_csc12_s0",
    ],
}


def expected_for(name):
    for key, n in PROTOCOL_EVALS:
        if key in name:
            return n
    return None


def inspect(path):
    n_eval = 0
    traceback = False
    last = ""
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "in csv format" in line:
                n_eval += 1
            elif "Traceback (most recent call last)" in line:
                traceback = True
            elif line.strip():
                last = line.rstrip()[:90]
    return n_eval, traceback, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--manifest", choices=sorted(MANIFESTS))
    ap.add_argument("--expect", nargs="*", default=[],
                    help="NAME=N overrides for the expected evaluation count")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.expect)
    found = {os.path.splitext(os.path.basename(p))[0]: p
             for p in sorted(glob.glob(os.path.join(args.logs, "*.log")))}

    names = list(MANIFESTS[args.manifest]) if args.manifest else []
    for n in sorted(found):
        if n not in names:
            names.append(n)

    rows = []
    for name in names:
        exp = int(overrides.get(name, expected_for(name) or 0)) or None
        path = found.get(name)
        if path is None:
            rows.append((name, "-", exp, "MISSING", "never run"))
            continue
        n_eval, tb, last = inspect(path)
        if tb:
            status = "FAILED"
        elif exp is None:
            # No known protocol: a final copypaste line means it exited cleanly.
            status = "OK*" if "copypaste:" in last else "UNKNOWN"
        elif n_eval >= exp:
            status = "OK"
        elif n_eval == 0:
            status = "FAILED"
        else:
            status = "INCOMPLETE"
        rows.append((name, n_eval, exp, status, last if status not in ("OK", "OK*") else ""))

    w = max(len(r[0]) for r in rows) + 2
    print(f"{'run':<{w}}{'evals':>7}{'exp':>6}  {'status':<11}note")
    print("-" * (w + 30))
    order = {"OK": 0, "OK*": 1, "INCOMPLETE": 2, "FAILED": 3, "MISSING": 4, "UNKNOWN": 5}
    for name, n_eval, exp, status, note in sorted(rows, key=lambda r: (order[r[3]], r[0])):
        print(f"{name:<{w}}{str(n_eval):>7}{str(exp or '?'):>6}  {status:<11}{note}")

    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    todo = [r[0] for r in rows if r[3] in ("MISSING", "FAILED", "INCOMPLETE")]
    if todo:
        print("\nNEEDS RUNNING / RERUNNING:")
        for t in todo:
            print("  " + t)


if __name__ == "__main__":
    main()
