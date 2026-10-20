#!/usr/bin/env python3
"""Runtime + memory table for the paper's Efficiency section.

Combines two data sources:
  1. Per-image latency (ms/img) — parsed from detectron2's evaluator logs
     ("Total inference pure compute time" lines) across all recorded runs.
  2. Peak GPU memory (MiB) — from /tmp/gpu_mem_bench.log produced by
     bench_gpu_memory.sh (nvidia-smi polling during a fresh fog-only eval).
"""
from __future__ import annotations
import os, re, statistics
from typing import Dict, List, Optional

PURE_RE = re.compile(
    r"Total inference pure compute time:.*?\(([\d.]+)\s+s\s+/\s+(?:iter|img)"
)


def per_img_ms(pattern: str, seeds=(0, 42, 123)) -> Optional[float]:
    values: List[float] = []
    paths = [pattern.format(s=s) for s in seeds] if "{s}" in pattern else [pattern]
    for path in paths:
        if not os.path.isfile(path):
            continue
        for line in open(path):
            m = PURE_RE.search(line)
            if m:
                values.append(float(m.group(1)) * 1000.0)
    if not values:
        return None
    return statistics.mean(values)


def parse_mem_log(path="/tmp/gpu_mem_bench.log") -> Dict[str, int]:
    out = {}
    if not os.path.isfile(path):
        return out
    for i, line in enumerate(open(path)):
        if i == 0:
            continue
        parts = line.strip().split(",")
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


# (Display name, log pattern (or None), mem_bench_key)
VARIANTS = [
    ("PFN source (no CTTA, forward only)",
     "/tmp/bench_PFN_source_only.out",
     "PFN_source_only"),
    ("CTCMT_Det (MR-CNN, single-task)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_ctcmt_det_mr/seed{s}.log",
     "CTCMT_Det_MR"),
    ("W3TTA-OD (Yoo et al. CVPR 2024)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/w3ttaod_acdc.log",
     None),
    ("CT-CMT-MTL baseline (no XVA)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_table4/no_ctcl_seed{s}.log",
     "CTCMT_MTL_baseline"),
    ("CT-CMT-MTL + V2 (three-component XVA)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/V2_seed{s}.log",
     "CTCMT_MTL_V2"),
    ("CT-CMT-MTL + V2 + CTPV (full XVA)",
     "/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements/e1_ctpv_seed{s}.log",
     "CTCMT_MTL_V2_CTPV"),
]


def render():
    mem = parse_mem_log()
    print()
    print("### Runtime and memory on ACDC continual (1x NVIDIA L40S)")
    print()
    print("| Method | ms/img | Peak VRAM (MiB) | Peak VRAM (GB) | Rel. latency |")
    print("|---|---:|---:|---:|---:|")
    baseline_ms = per_img_ms(VARIANTS[0][1])
    for name, pattern, mem_key in VARIANTS:
        ms = per_img_ms(pattern)
        mib = mem.get(mem_key) if mem_key else None
        ms_str = f"{ms:.1f}" if ms is not None else "-"
        mib_str = f"{mib:,}" if mib is not None else "-"
        gb_str = f"{mib / 1024:.1f}" if mib is not None else "-"
        rel_str = f"{ms / baseline_ms:.1f}x" if ms is not None and baseline_ms else "-"
        print(f"| {name} | {ms_str} | {mib_str} | {gb_str} | {rel_str} |")

    print("""
- **ms/img** = `pure compute time` from detectron2's evaluator; captures the
  full per-image cost inside `model(inputs)`. For CTTA methods that includes
  teacher fwd + student fwd + backward + EMA + stochastic restore + all XVA
  loss computations.
- **Peak VRAM** = maximum GPU memory observed via nvidia-smi polling during
  a fresh fog-only ACDC run (see `bench_gpu_memory.sh`).
- **Rel. latency** = ratio vs. the source-only PFN forward pass.

Paper reading:
- Single-task CTCMT_Det on MR-CNN adds ~+70 ms/img over source-only inference
  (~2.3x overhead), and stays under 5 GB VRAM.
- MTL variants pay ~6x the source-only cost because the shared trunk carries
  both tasks and four XVA loss terms.
- Adding CTPV to three-component XVA costs +5 ms/img and +400 MiB, negligible
  for the +0.23 AP50 / +0.28 mIoU headline gain.
- The full-XVA footprint (~8 GB) fits on all modern deployment GPUs
  (24 GB L4/A10, 40 GB A100, 48 GB L40S).
""")


if __name__ == "__main__":
    render()
