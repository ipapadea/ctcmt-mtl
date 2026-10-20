#!/usr/bin/env python3
"""
Gather current CT-CR A/A2/B/C experiment results from tee logs.

Usage:
  python3 gather_ctcr_current.py
  python3 gather_ctcr_current.py --log-root /home/ilias
  python3 gather_ctcr_current.py --include-partial

By default:
- scans ~/ctcr_*.log plus any explicit --log-root
- recognizes A, A2, B, C
- recognizes ACDC, Cityscapes-C mixed LT x10, fog x10
- marks runs COMPLETE only when the expected number of bbox AND sem_seg evals exists
- prints complete runs separately from partial/running runs
- writes one CSV per host
"""

from __future__ import annotations

import argparse
import csv
import re
import socket
from pathlib import Path
from statistics import mean

VARIANT_PATTERNS = [
    ("A2", "ctcr_A2_v2_perbox_full_no_ctpv"),
    ("A",  "ctcr_A_v2_full_no_ctpv"),
    ("B",  "ctcr_B_v2_hard_t03_no_ctpv"),
    ("C",  "ctcr_C_v2_soft_a02_no_ctpv"),
]

BENCHMARKS = {
    "csc_mixed_lt_x10": {
        "needle": "_csc_mixed_lt_x10_",
        "expected": 50,
        "domains": ["fog", "motion_blur", "snow", "brightness", "defocus_blur"],
    },
    "acdc": {
        "needle": "_acdc_seed",
        "expected": 4,
        "domains": ["fog", "night", "rain", "snow"],
    },
    "csc_fog_x10": {
        "needle": "_fogx10",
        "expected": 10,
        "domains": ["fog"],
    },
}


def extract_metrics(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    ap50, miou = [], []

    for i, line in enumerate(lines):
        if "copypaste: Task: bbox" in line:
            # Detectron2 CSV block:
            # copypaste: Task: bbox
            # copypaste: AP,AP50,...
            # copypaste: x,y,...
            if i + 2 < len(lines):
                try:
                    vals = lines[i + 2].split("copypaste:", 1)[1].strip().split(",")
                    ap50.append(float(vals[1]))
                except Exception:
                    pass

        if "copypaste: Task: sem_seg" in line:
            # SemSeg CSV block: first reported metric is mIoU.
            if i + 2 < len(lines):
                try:
                    vals = lines[i + 2].split("copypaste:", 1)[1].strip().split(",")
                    miou.append(float(vals[0]))
                except Exception:
                    pass

    return ap50, miou


def get_variant(name: str):
    # A2 MUST be checked before A.
    for variant, needle in VARIANT_PATTERNS:
        if needle in name:
            return variant
    return None


def get_benchmark(name: str):
    for bench, info in BENCHMARKS.items():
        if info["needle"] in name:
            return bench
    return None


def get_seed(name: str):
    m = re.search(r"_seed(\d+)", name)
    return int(m.group(1)) if m else None


def hardware_note(name: str):
    if "l40scheck" in name.lower():
        return "L40S correction"
    return ""


def discover_logs(roots):
    found = {}
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue

        # tee logs created by the runners
        for p in root.glob("ctcr_*.log"):
            found[str(p.resolve())] = p

        # Also accept Detectron2 log.txt under experiment directories.
        # Restrict to paths containing ctcr_ to avoid scanning unrelated runs.
        for p in root.glob("**/log.txt"):
            if "ctcr_" in str(p):
                found[str(p.resolve())] = p

    return sorted(found.values())


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        action="append",
        default=[],
        help="Directory to scan. Can be given more than once.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include partial runs in the detail CSV summaries.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Where to write CSVs (default: current directory).",
    )
    args = parser.parse_args()

    roots = [Path.home()]
    roots += [Path(x) for x in args.log_root]

    logs = discover_logs(roots)
    host = socket.gethostname()

    runs = []
    seen_signature = set()

    for p in logs:
        name = p.parent.name if p.name == "log.txt" else p.name

        variant = get_variant(name)
        bench = get_benchmark(name)
        seed = get_seed(name)

        if variant is None or bench is None or seed is None:
            continue

        ap50, miou = extract_metrics(p)
        info = BENCHMARKS[bench]
        expected = info["expected"]

        # Prefer the source with the larger number of parsed evals if both
        # tee log and Detectron2 log.txt represent the same run.
        sig = (variant, bench, seed, hardware_note(name))
        record = {
            "host": host,
            "variant": variant,
            "benchmark": bench,
            "seed": seed,
            "hardware_note": hardware_note(name),
            "path": str(p),
            "expected": expected,
            "n_ap50": len(ap50),
            "n_miou": len(miou),
            "complete": len(ap50) == expected and len(miou) == expected,
            "ap50": ap50,
            "miou": miou,
        }

        prev_idx = None
        for idx, r in enumerate(runs):
            if (r["variant"], r["benchmark"], r["seed"], r["hardware_note"]) == sig:
                prev_idx = idx
                break

        if prev_idx is None:
            runs.append(record)
        else:
            old = runs[prev_idx]
            if min(len(ap50), len(miou)) > min(old["n_ap50"], old["n_miou"]):
                runs[prev_idx] = record

    runs.sort(key=lambda r: (r["benchmark"], r["variant"], r["seed"], r["hardware_note"]))

    print("=" * 108)
    print(f"HOST: {host}")
    print("=" * 108)

    if not runs:
        print("No recognized CT-CR logs found.")
        print("Scanned:")
        for r in roots:
            print(" ", r)
        return

    print("\nRUN STATUS")
    print(f"{'Variant':<8} {'Benchmark':<20} {'Seed':<6} {'bbox':<8} {'mIoU':<8} {'Status':<10} {'Note'}")
    print("-" * 108)
    for r in runs:
        status = "COMPLETE" if r["complete"] else "PARTIAL"
        print(
            f"{r['variant']:<8} {r['benchmark']:<20} {r['seed']:<6} "
            f"{r['n_ap50']:>2}/{r['expected']:<4} {r['n_miou']:>2}/{r['expected']:<4} "
            f"{status:<10} {r['hardware_note']}"
        )

    print("\nCOMPLETE RUN MEANS")
    print(f"{'Variant':<8} {'Benchmark':<20} {'Seed':<6} {'Mean AP50':<14} {'Mean mIoU':<14} {'Note'}")
    print("-" * 108)

    complete_runs = [r for r in runs if r["complete"]]
    for r in complete_runs:
        print(
            f"{r['variant']:<8} {r['benchmark']:<20} {r['seed']:<6} "
            f"{mean(r['ap50']):<14.4f} {mean(r['miou']):<14.4f} {r['hardware_note']}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"ctcr_run_summary_{host}.csv"
    with summary_path.open("w", newline="") as f:
        fields = [
            "host", "variant", "benchmark", "seed", "hardware_note",
            "status", "expected_evals", "n_ap50", "n_miou",
            "mean_ap50", "mean_miou", "first_ap50", "last_ap50",
            "first_miou", "last_miou", "path",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in runs:
            if not r["complete"] and not args.include_partial:
                continue
            aps, mis = r["ap50"], r["miou"]
            w.writerow({
                "host": r["host"],
                "variant": r["variant"],
                "benchmark": r["benchmark"],
                "seed": r["seed"],
                "hardware_note": r["hardware_note"],
                "status": "COMPLETE" if r["complete"] else "PARTIAL",
                "expected_evals": r["expected"],
                "n_ap50": r["n_ap50"],
                "n_miou": r["n_miou"],
                "mean_ap50": mean(aps) if aps else "",
                "mean_miou": mean(mis) if mis else "",
                "first_ap50": aps[0] if aps else "",
                "last_ap50": aps[-1] if aps else "",
                "first_miou": mis[0] if mis else "",
                "last_miou": mis[-1] if mis else "",
                "path": r["path"],
            })

    detail_path = out_dir / f"ctcr_eval_detail_{host}.csv"
    with detail_path.open("w", newline="") as f:
        fields = [
            "host", "variant", "benchmark", "seed", "hardware_note",
            "eval_idx", "round", "domain", "ap50", "miou", "path",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in runs:
            if not r["complete"] and not args.include_partial:
                continue

            n = min(len(r["ap50"]), len(r["miou"]))
            domains = BENCHMARKS[r["benchmark"]]["domains"]

            for i in range(n):
                if r["benchmark"] == "csc_mixed_lt_x10":
                    round_idx = i // len(domains) + 1
                elif r["benchmark"] == "csc_fog_x10":
                    round_idx = i + 1
                else:
                    round_idx = 1

                w.writerow({
                    "host": r["host"],
                    "variant": r["variant"],
                    "benchmark": r["benchmark"],
                    "seed": r["seed"],
                    "hardware_note": r["hardware_note"],
                    "eval_idx": i + 1,
                    "round": round_idx,
                    "domain": domains[i % len(domains)],
                    "ap50": r["ap50"][i],
                    "miou": r["miou"][i],
                    "path": r["path"],
                })

    print("\nWROTE")
    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()
