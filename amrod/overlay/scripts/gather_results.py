#!/usr/bin/env python3
"""Collect every CTTA run into per-protocol result tables.

Reads the detectron2 logs directly: each one contains the fully resolved config
("Running with full config:"), so metrics are always tied to the hyperparameters
that actually produced them rather than to a filename convention.

Outputs, under --out (default results/collected/):
  runs.csv              one row per run: metadata + hyperparameters + summary
  per_eval.csv          one row per evaluation (per domain, per round)
  tables/<PROTOCOL>.md  per-protocol markdown table
  tables/<PROTOCOL>_seedavg.md   same, averaged over seeds (mean +/- sample std)

Usage:
  python3 scripts/gather_results.py --logs DIR [DIR ...]
  python3 scripts/gather_results.py --logs local/logs remote/logs --out results/collected
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

# Hyperparameters surfaced as table columns. Order matters for readability.
HYPERS = [
    ("arch", r"^\s*META_ARCHITECTURE:\s*(\S+)"),
    ("source", r"^\s*WEIGHTS:\s*\S*/([^/]+)/model_final\.pth"),
    ("lr", r"^\s*BASE_LR:\s*(\S+)"),
    ("mt", r"^\s*MT:\s*(\S+)"),
    ("rst", r"^\s*RST_M:\s*(\S+)"),
    ("strong_aug", r"^\s*CTCMT_STRONG_AUG_STUDENT:\s*(\S+)"),
    ("ctcr_mode", r"^\s*CTCMT_CTCR_MODE:\s*(\S+)"),
    ("w_ctcr", r"^\s*CTCMT_WEIGHT_CTCR:\s*(\S+)"),
    ("w_ctcl", r"^\s*CTCMT_WEIGHT_CTCL:\s*(\S+)"),
    ("cls_bal", r"^\s*CTCMT_CLASS_BALANCED_CE:\s*(\S+)"),
    ("scale_pres", r"^\s*CTCMT_SEG_LOSS_SCALE_PRESERVE:\s*(\S+)"),
    ("anchor_marg", r"^\s*CTCMT_ANCHOR_MARGINAL_WEIGHT:\s*(\S+)"),
    ("fisher_rst", r"^\s*CTCMT_FISHER_RESTORE:\s*(\S+)"),
    ("v2_backbone", r"^\s*CTCMT_CROSS_TASK_FISHER:\s*(\S+)"),
    ("ctpv", r"^\s*CTCMT_CTPV_ENABLED:\s*(\S+)"),
    ("seed", r"^\s*SEED:\s*(\S+)"),
]

CORRUPTIONS = {
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost",
    "fog", "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "gaussian_noise", "shot_noise", "impulse_noise",
}


def parse_log(path):
    lines = open(path, errors="ignore").read().splitlines()

    start = next((i for i, l in enumerate(lines) if "Running with full config" in l), None)
    cfg_region = []
    if start is not None:
        for l in lines[start + 1:]:
            if re.match(r"^\[\d\d/\d\d ", l):
                break
            cfg_region.append(l)

    meta = {}
    for name, pat in HYPERS:
        rx = re.compile(pat)
        val = ""
        for l in cfg_region:
            m = rx.match(l)
            if m:
                val = m.group(1)
                break
        meta[name] = val

    m = re.search(r"config_file='([^']+)'", "\n".join(lines[:200]))
    meta["config"] = os.path.basename(m.group(1)) if m else ""

    evals, ds = [], None
    for i, l in enumerate(lines):
        m = re.search(r"Evaluation results for (\S+) in csv format", l)
        if m:
            ds = m.group(1)
            evals.append({"dataset": ds, "AP50": None, "AP": None, "mIoU": None})
        if ds and "copypaste: Task:" in l and i + 2 < len(lines):
            task = l.split("Task:")[1].strip()
            try:
                hdr = lines[i + 1].split("copypaste:")[1].strip().split(",")
                val = [float(x) for x in lines[i + 2].split("copypaste:")[1].strip().split(",")]
            except (IndexError, ValueError):
                continue
            d = dict(zip(hdr, val))
            if task == "bbox":
                evals[-1]["AP50"] = d.get("AP50")
                evals[-1]["AP"] = d.get("AP")
            elif task == "sem_seg":
                evals[-1]["mIoU"] = d.get("mIoU")
    return meta, evals


def protocol_of(evals):
    names = [e["dataset"].replace("_mtl", "").replace("_semseg", "") for e in evals]
    n = len(names)
    if not n:
        return "EMPTY"
    if all(x.startswith("acdc") for x in names):
        return "ACDC-LT-x10" if n >= 40 else ("ACDC-4dom" if n == 4 else f"ACDC-{n}evals")
    if all(x in CORRUPTIONS for x in names):
        uniq = len(set(names))
        if uniq == 1:
            return f"CSC-{names[0]}-x{n}"
        if uniq == 5 and n >= 50:
            return "CSC-LT-x10"
        if n == 12:
            return "CSC-12corr"
        return f"CSC-{uniq}dom-{n}evals"
    return f"other-{n}evals"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, float) else ("" if v is None else str(v))


def summarise(evals, protocol):
    ap = [e["AP50"] for e in evals]
    iu = [e["mIoU"] for e in evals]
    row = {"n_evals": len(evals), "AP50": mean(ap), "mIoU": mean(iu),
           "R1_AP50": None, "R10_AP50": None, "R1_mIoU": None, "R10_mIoU": None}
    per_round = {"ACDC-LT-x10": 4, "CSC-LT-x10": 5}.get(protocol)
    if per_round and len(evals) >= per_round * 2:
        row["R1_AP50"] = mean(ap[:per_round])
        row["R10_AP50"] = mean(ap[-per_round:])
        row["R1_mIoU"] = mean(iu[:per_round])
        row["R10_mIoU"] = mean(iu[-per_round:])
    return row


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--logs", nargs="+", required=True)
    ap_.add_argument("--out", default="results/collected")
    ap_.add_argument("--min-evals", type=int, default=1)
    args = ap_.parse_args()

    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    runs, per_eval = [], []
    for d in args.logs:
        for path in sorted(glob.glob(os.path.join(d, "*.log"))):
            if path.endswith(".preflight.log"):
                continue
            meta, evals = parse_log(path)
            if len(evals) < args.min_evals:
                continue
            name = os.path.basename(path)[:-4]
            proto = protocol_of(evals)
            row = {"run": name, "protocol": proto, "logdir": d}
            row.update(meta)
            row.update(summarise(evals, proto))
            runs.append(row)
            for i, e in enumerate(evals):
                per_eval.append({"run": name, "protocol": proto, "idx": i, **e})

    if not runs:
        print("no runs found")
        return

    cols = (["run", "protocol"] + [h for h, _ in HYPERS] + ["config", "n_evals"]
            + ["AP50", "mIoU", "R1_AP50", "R10_AP50", "R1_mIoU", "R10_mIoU", "logdir"])
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(runs, key=lambda r: (r["protocol"], r["run"])):
            w.writerow(r)
    with open(os.path.join(args.out, "per_eval.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "protocol", "idx", "dataset", "AP50", "AP", "mIoU"])
        w.writeheader()
        w.writerows(per_eval)

    # Hyperparameters that actually vary are the ones worth showing.
    by_proto = defaultdict(list)
    for r in runs:
        by_proto[r["protocol"]].append(r)

    for proto, rows in sorted(by_proto.items()):
        varying = [h for h, _ in HYPERS
                   if h not in ("seed", "arch", "source")
                   and len({r.get(h, "") for r in rows}) > 1]
        show = ["run", "arch", "source"] + varying + ["seed", "n_evals",
                "AP50", "mIoU", "R1_AP50", "R10_AP50", "R1_mIoU", "R10_mIoU"]
        path = os.path.join(args.out, "tables", f"{proto}.md")
        with open(path, "w") as fh:
            fh.write(f"# {proto}\n\n({len(rows)} runs)\n\n")
            fh.write("| " + " | ".join(show) + " |\n")
            fh.write("|" + "---|" * len(show) + "\n")
            for r in sorted(rows, key=lambda r: -(r["AP50"] or r["mIoU"] or 0)):
                fh.write("| " + " | ".join(fmt(r.get(c)) for c in show) + " |\n")

        # seed-averaged view: group runs that share every varying hyperparameter
        groups = defaultdict(list)
        for r in rows:
            key = tuple(r.get(h, "") for h in ["arch", "source"] + varying)
            groups[key].append(r)
        path = os.path.join(args.out, "tables", f"{proto}_seedavg.md")
        with open(path, "w") as fh:
            head = ["arch", "source"] + varying + ["n", "AP50", "mIoU", "R10_AP50", "R10_mIoU"]
            fh.write(f"# {proto} — averaged over seeds\n\n")
            fh.write("| " + " | ".join(head) + " |\n")
            fh.write("|" + "---|" * len(head) + "\n")
            for key, g in sorted(groups.items(),
                                 key=lambda kv: -(mean([r["AP50"] for r in kv[1]])
                                                  or mean([r["mIoU"] for r in kv[1]]) or 0)):
                cells = list(key) + [str(len(g))]
                for m in ["AP50", "mIoU", "R10_AP50", "R10_mIoU"]:
                    mu, sd = mean([r[m] for r in g]), std([r[m] for r in g])
                    cells.append("" if mu is None else
                                 (f"{mu:.3f}" if sd is None else f"{mu:.3f} ± {sd:.3f}"))
                fh.write("| " + " | ".join(cells) + " |\n")

    print(f"{len(runs)} runs -> {args.out}/runs.csv, per_eval.csv")
    for proto, rows in sorted(by_proto.items()):
        print(f"  {proto:<18} {len(rows):>3} runs -> tables/{proto}.md (+ _seedavg)")


if __name__ == "__main__":
    main()
