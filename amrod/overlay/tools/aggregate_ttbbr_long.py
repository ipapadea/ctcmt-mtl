"""Aggregate long-term CTTA TT-BBR sweep (paper Table 3 protocol).

Parses `stdout.log` from each run in /data/vgcmt/work_dirs/ttbbr_long/
(5 seeds: 1 unseeded + 4 fixed), extracts:
  - per-round-per-corruption AP50 (5 corr * 10 rounds = 50 datapoints)
  - AP50_ALL_mean (single scalar)
  - selected rounds table (round 1, 5, 10) matching paper's presentation

Produces:
  1. Per-seed mean AP50 sorted.
  2. Multi-seed mean +/- std (paper's variance report).
  3. Round-by-round evolution (mean over corruptions per round).
  4. Comparison table vs AMROD paper Table 3.
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

LONG_ROOT = Path("/data/vgcmt/work_dirs/ttbbr_long")

# Corruption order in the long-term protocol (matches cfg_cityscapes_c_long.yaml).
LT_CORR = ["fog", "motion_blur", "snow", "brightness", "defocus_blur"]
SHORT = ["Fog", "Mot", "Snow", "Brt", "Def"]

# Paper Table 3 reference numbers (mAP@50 mean across all rounds).
PAPER_TABLE3 = {
    "Source": 16.4,
    "Tent": 12.0,
    "CoTTA": 19.1,
    "WHW": 24.2,
    "SVDP": 25.3,
    "IRG": 25.6,
    "MemCLR": 26.0,
    "AMROD-unstop": 29.0,
    "AMROD": 29.2,
}


def parse_long_log(log_path: Path) -> dict | None:
    """Extract per-round-per-corruption AP50 and overall mean.

    Returns dict:
        {
            "per_round_per_corr": {round: {corr: AP50, ...}, ...},
            "mean": float,
        }
    or None if incomplete.
    """
    if not log_path.is_file():
        return None
    text = log_path.read_text()

    per_round: dict[int, dict[str, float]] = {}
    for m in re.finditer(
        r"round:\s*(\d+)\s+dataset:\s+(\w+)\s+AP50:\s*([\d.]+)", text
    ):
        r = int(m.group(1))
        c = m.group(2)
        v = float(m.group(3))
        per_round.setdefault(r, {})[c] = v

    mean_match = re.search(r"AP50_ALL_mean:\s*([\d.]+)", text)
    if mean_match is None or not per_round:
        return None
    # Sanity: expect 10 rounds x 5 corr = 50 measurements.
    return {
        "per_round_per_corr": per_round,
        "mean": float(mean_match.group(1)),
        "n_rounds": len(per_round),
        "n_corr": len(next(iter(per_round.values()))),
    }


def find_log(run_dir: Path) -> Path | None:
    for cand in ("stdout.log", "log.txt"):
        p = run_dir / cand
        if p.is_file():
            with open(p) as f:
                if "AP50_ALL_mean:" in f.read():
                    return p
    return None


def load_runs() -> list[dict]:
    runs = []
    if not LONG_ROOT.is_dir():
        return runs
    for d in sorted(LONG_ROOT.iterdir()):
        if not d.is_dir():
            continue
        log = find_log(d)
        if log is None:
            print(f"[aggregate-long] SKIP {d.name} (incomplete)")
            continue
        data = parse_long_log(log)
        if data is None:
            continue
        runs.append({"name": d.name, **data})
    return runs


def print_seeds(runs):
    print("\n" + "=" * 100)
    print("Long-term TT-BBR runs (per-seed mean AP50, sorted)")
    print("=" * 100)
    print(f"{'seed name':<12} {'rounds':>7} {'corr':>5} {'mean AP50':>10}")
    print("-" * 40)
    for r in sorted(runs, key=lambda x: -x["mean"]):
        print(f"{r['name']:<12} {r['n_rounds']:>7} {r['n_corr']:>5} "
              f"{r['mean']:>10.2f}")


def print_variance(runs):
    if not runs:
        return
    print("\n" + "=" * 100)
    print("Multi-seed variance (5 seeds)")
    print("=" * 100)
    means = [r["mean"] for r in runs]
    mn = statistics.mean(means)
    sd = statistics.stdev(means) if len(means) > 1 else 0.0
    print(f"AP50_ALL_mean across {len(means)} seeds:")
    print(f"    mean = {mn:.2f}  +/-  {sd:.2f}")
    print(f"    min  = {min(means):.2f}")
    print(f"    max  = {max(means):.2f}")


def print_per_round(runs):
    if not runs:
        return
    print("\n" + "=" * 100)
    print("Round-by-round evolution (mean over the 5 corruptions), averaged across seeds")
    print("=" * 100)
    # Aggregate: per round, mean over corr, then mean across seeds.
    n_rounds = min(r["n_rounds"] for r in runs)
    print(f"{'round':>6} {'mean AP50':>10} {'std':>7}")
    for rd in range(1, n_rounds + 1):
        per_seed_round = []
        for r in runs:
            corr_dict = r["per_round_per_corr"].get(rd, {})
            if corr_dict:
                per_seed_round.append(sum(corr_dict.values()) / len(corr_dict))
        if per_seed_round:
            m = statistics.mean(per_seed_round)
            s = statistics.stdev(per_seed_round) if len(per_seed_round) > 1 else 0.0
            print(f"{rd:>6} {m:>10.2f} {s:>7.2f}")


def print_selected_rounds(runs):
    """Match paper's Table 3 presentation: rounds 1, 5, 10."""
    if not runs:
        return
    print("\n" + "=" * 100)
    print("Selected rounds (1, 5, 10) -- per-corr mean across seeds")
    print("=" * 100)
    hdr = f"{'round':>6} " + "  ".join(f"{s:>7}" for s in SHORT) + f"  {'mean':>7}"
    print(hdr)
    print("-" * len(hdr))
    for rd in (1, 5, 10):
        per_seed_corr = {c: [] for c in LT_CORR}
        for r in runs:
            corr_dict = r["per_round_per_corr"].get(rd, {})
            for c in LT_CORR:
                if c in corr_dict:
                    per_seed_corr[c].append(corr_dict[c])
        row_vals = []
        for c in LT_CORR:
            if per_seed_corr[c]:
                row_vals.append(statistics.mean(per_seed_corr[c]))
            else:
                row_vals.append(float("nan"))
        if any(v != v for v in row_vals):
            print(f"{rd:>6}  (incomplete)")
            continue
        mn = statistics.mean(row_vals)
        print(f"{rd:>6} " + "  ".join(f"{v:>7.2f}" for v in row_vals) + f"  {mn:>7.2f}")


def print_paper_comparison(runs):
    if not runs:
        return
    print("\n" + "=" * 100)
    print("Comparison vs AMROD paper Table 3 (mAP@50 mean across all rounds)")
    print("=" * 100)
    means = [r["mean"] for r in runs]
    ours = statistics.mean(means)
    ours_sd = statistics.stdev(means) if len(means) > 1 else 0.0

    rows = list(PAPER_TABLE3.items())
    rows.append((f"OURS (AMROD + TT-BBR)", ours))
    rows.sort(key=lambda x: x[1])

    print(f"{'method':<25} {'mean AP50':>10}")
    print("-" * 40)
    for name, v in rows:
        marker = "  <-- ours" if name == "OURS (AMROD + TT-BBR)" else ""
        star_sd = f"  +/- {ours_sd:.2f}" if name == "OURS (AMROD + TT-BBR)" else ""
        print(f"{name:<25} {v:>10.2f}{star_sd}{marker}")

    amrod_paper = PAPER_TABLE3["AMROD"]
    delta = ours - amrod_paper
    print()
    print(f"Delta vs AMROD paper 29.2 : {delta:+.2f} mAP@50")
    if delta > 0:
        print("We BEAT AMROD's published long-term result.")
    elif delta > -0.5:
        print("Within noise of AMROD.")
    else:
        print("Below AMROD; consider tuning TT-BBR IoU threshold for long-term.")


def main():
    runs = load_runs()
    if not runs:
        print("[aggregate-long] no completed long-term runs found under "
              f"{LONG_ROOT}")
        return
    print_seeds(runs)
    print_variance(runs)
    print_per_round(runs)
    print_selected_rounds(runs)
    print_paper_comparison(runs)


if __name__ == "__main__":
    main()
