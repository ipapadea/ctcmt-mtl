#!/usr/bin/env python3
"""Extract enhancement-matrix (E1..E6) multi-seed results.

Compares 5 individual enhancements + kitchen-sink E6 against the baseline
CTCMT-MTL+V2 numbers (already known from logs_multiseed/V2_seed*.log).
"""
import os, re, statistics

HDR_RE = re.compile(r"Evaluation results for (\S+) in csv format")
TASK_RE = re.compile(r"copypaste: Task: (\S+)")
VALS_RE = re.compile(r"copypaste: ([\d.,]+)$")
DOMAINS = ["fog", "night", "rain", "snow"]
SEEDS = [0, 42, 123]


def parse_log(path):
    out = {d: [None, None] for d in DOMAINS}  # [mIoU, AP50]
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
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


VARIANTS = [
    ("Baseline (CTCMT-MTL + V2)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/V2_seed{s}.log"),
    ("E1  CTPV",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements/e1_ctpv_seed{s}.log"),
    ("E2  Entropy-weighted CE",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements/e2_entropy_ce_seed{s}.log"),
    ("E3  Teacher-triggered aug-avg",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements/e3_teacher_aug_seed{s}.log"),
    ("E4  Directional gate (boost=2.0)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements/e4_dir_gate_seed{s}.log"),
    ("E4b Directional gate (boost=1.5)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements2/e4b_seed{s}.log"),
    ("E5  Adaptive STR",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements/e5_adaptive_str_seed{s}.log"),
    ("E6  Kitchen sink (all)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements/e6_all_seed{s}.log"),
    ("E7  E1 + E3 (winners only)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements2/e7_seed{s}.log"),
]


def collect():
    return {name: [parse_log(pattern.format(s=s)) for s in SEEDS]
            for name, pattern in VARIANTS}


def render(table, metric_idx, label):
    print(f"\n### {label}")
    print("| Variant | Fog | Night | Rain | Snow | **Mean** |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, per_seed in table.items():
        row = [name]
        for dom in DOMAINS:
            row.append(fmt([ps[dom][metric_idx] for ps in per_seed]))
        seed_means = [sum(ps[d][metric_idx] for d in DOMAINS) / 4
                      for ps in per_seed if all(ps[d][metric_idx] is not None for d in DOMAINS)]
        row.append(fmt(seed_means))
        print("| " + " | ".join(row) + " |")


def main():
    table = collect()
    render(table, 1, "Enhancement matrix — Detection (bbox AP50)")
    render(table, 0, "Enhancement matrix — Segmentation (mIoU)")


if __name__ == "__main__":
    main()
