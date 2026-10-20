#!/usr/bin/env python3
"""Extract Table 4 multi-seed results and print mean+/-std tables.

Parses the copypaste: markers in the log tail per (dataset, task) and computes
mean+/-std across seeds for each config x domain x metric.

Configs covered:
- MTL no-CL no-V2   (logs_table4/no_ctcl_seed{0,42,123}.log)
- MTL no-CL + V2    (logs_table4/no_ctcl_v2_seed{0,42,123}.log)
- MTL CT-CL no-V2   (logs_multiseed/ctcl_noV2_seed{0,42,123}.log)   [already done]
- CTCMT_MTL + V2    (logs_multiseed/V2_seed{0,42,123}.log)          [already done]
"""
import os, re, statistics, argparse

HDR_RE = re.compile(r"Evaluation results for (\S+) in csv format")
TASK_RE = re.compile(r"copypaste: Task: (\S+)")
VALS_RE = re.compile(r"copypaste: ([\d.,]+)$")
DOMAINS = ["fog", "night", "rain", "snow"]


def parse_log(path):
    """Returns dict[domain] -> (mIoU, AP50) or None if missing."""
    out = {d: [None, None] for d in DOMAINS}
    if not os.path.isfile(path):
        return out
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        m = HDR_RE.search(line)
        if not m:
            continue
        ds = m.group(1)
        dom = next((d for d in DOMAINS if d in ds), None)
        if dom is None:
            continue
        for j in range(i + 1, min(i + 15, len(lines))):
            mt = TASK_RE.search(lines[j])
            if mt and j + 2 < len(lines):
                mv = VALS_RE.search(lines[j + 2])
                if not mv:
                    continue
                vals = mv.group(1).split(",")
                task = mt.group(1)
                if task == "sem_seg":
                    out[dom][0] = float(vals[0])
                elif task == "bbox":
                    out[dom][1] = float(vals[1])
    return out


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return f"{vals[0]:.2f}" if vals else "—"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


CONFIGS = [
    ("MTL no-CL no-V2",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_table4/no_ctcl_seed{seed}.log"),
    ("MTL no-CL + V2",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_table4/no_ctcl_v2_seed{seed}.log"),
    ("MTL CT-CL no-V2",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/ctcl_noV2_seed{seed}.log"),
    ("CTCMT_MTL + V2",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/V2_seed{seed}.log"),
]
SEEDS = [0, 42, 123]


def collect():
    table = {}
    for name, pattern in CONFIGS:
        per_seed = [parse_log(pattern.format(seed=s)) for s in SEEDS]
        table[name] = per_seed
    return table


def render(table, metric_idx, label):
    print(f"\n### {label}")
    print("| Config | Fog | Night | Rain | Snow | **Mean** |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, per_seed in table.items():
        row = [name]
        domain_means = []
        for dom in DOMAINS:
            vals = [ps[dom][metric_idx] for ps in per_seed]
            row.append(fmt(vals))
            valid = [v for v in vals if v is not None]
            if valid:
                domain_means.append(statistics.mean(valid))
        seed_means = []
        for s_idx in range(len(SEEDS)):
            per_dom = [per_seed[s_idx][d][metric_idx] for d in DOMAINS]
            if all(v is not None for v in per_dom):
                seed_means.append(sum(per_dom) / 4)
        row.append(fmt(seed_means))
        print("| " + " | ".join(row) + " |")


def main():
    table = collect()
    render(table, 1, "Table 4 — Detection (bbox AP50)")
    render(table, 0, "Table 4 — Segmentation (mIoU)")


if __name__ == "__main__":
    main()
