"""Aggregate TT-BBR sweep results into a paper-ready table.

Reads all `/data/vgcmt/work_dirs/ttbbr_sweep/*/log.txt` files, parses
per-corruption AP50 and `AP50_ALL_mean`, produces:
  1. Summary of ALL runs sorted by mean.
  2. Phase 1 (IoU sweep) as a line-per-IoU table.
  3. Phase 3 (multi-seed) as mean +/- std at the chosen config.
  4. Best-config recommendation.
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

SWEEP_ROOT = Path("/data/vgcmt/work_dirs/ttbbr_sweep")
# The original TT-BBR run (random seed 2501266, iou=0.7, drop=False) that
# produced the initial 21.8 mAP@50. Included as the "5th seed" for the
# multi-seed variance report per the user's request.
ORIGINAL_TTBBR_DIR = Path("/data/vgcmt/work_dirs/amrod_ttbbr")
ORIGINAL_TTBBR_META = dict(iou=0.70, drop=False, seed=2501266,
                            name="iou0.70_drop0_seedRAND")

CORR_ORDER = [
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
SHORT = ["Def", "Gls", "Mot", "Zoom", "Snow", "Frst", "Fog", "Brt",
         "Ctr", "Ela", "Pix", "JPG"]

RUN_RE = re.compile(r"^iou(?P<iou>[\d.]+)_drop(?P<drop>[01])_seed(?P<seed>\d+)$")


def parse_log(log_path: Path) -> dict | None:
    """Return dict with per-corruption AP50 and mean, or None if incomplete."""
    if not log_path.is_file():
        return None
    text = log_path.read_text()
    per_corr = {}
    for c in CORR_ORDER:
        m = re.search(rf"round: 1 dataset: {c} AP50: ([\d.]+)", text)
        if m:
            per_corr[c] = float(m.group(1))
    mean_match = re.search(r"AP50_ALL_mean: ([\d.]+)", text)
    if mean_match is None or len(per_corr) != 12:
        return None
    return {
        "per_corruption": per_corr,
        "mean": float(mean_match.group(1)),
    }


def find_run_log(run_dir: Path) -> Path | None:
    """Locate the log file inside a run dir. Prefer stdout.log (contains
    AP50_ALL_mean from adapt.py's stdout); fall back to log.txt (D2's own
    logger, may or may not include AP50_ALL_mean depending on setup)."""
    for cand in ("stdout.log", "log.txt"):
        p = run_dir / cand
        if p.is_file():
            with open(p) as f:
                if "AP50_ALL_mean:" in f.read():
                    return p
    return None


def load_all_runs() -> list[dict]:
    runs = []
    # Sweep-directory runs
    for run_dir in sorted(SWEEP_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        log = find_run_log(run_dir)
        if log is None:
            print(f"[aggregate] SKIP {run_dir.name} (incomplete)")
            continue
        data = parse_log(log)
        if data is None:
            print(f"[aggregate] SKIP {run_dir.name} (parse failed)")
            continue
        runs.append({
            "name": run_dir.name,
            "iou": float(m.group("iou")),
            "drop": bool(int(m.group("drop"))),
            "seed": int(m.group("seed")),
            "mean": data["mean"],
            "per_corruption": data["per_corruption"],
        })

    # Original un-seeded TT-BBR run: pull in as the 5th seed for the variance
    # report at iou=0.7 drop=False.
    orig_log = find_run_log(ORIGINAL_TTBBR_DIR)
    if orig_log is not None:
        orig_data = parse_log(orig_log)
        if orig_data is not None:
            runs.append({
                **ORIGINAL_TTBBR_META,
                "mean": orig_data["mean"],
                "per_corruption": orig_data["per_corruption"],
            })
        else:
            print(f"[aggregate] SKIP original amrod_ttbbr (parse failed)")
    else:
        print(f"[aggregate] SKIP original amrod_ttbbr (no log with AP50_ALL_mean)")

    return runs


def print_all(runs):
    print("\n" + "=" * 100)
    print("All sweep runs (sorted by mean AP50)")
    print("=" * 100)
    print(f"{'name':<28} {'iou':>5} {'drop':>5} {'seed':>5} "
          + " ".join(f"{s:>5}" for s in SHORT) + f" {'MEAN':>6}")
    for r in sorted(runs, key=lambda x: -x["mean"]):
        vals = [r["per_corruption"][c] for c in CORR_ORDER]
        print(f"{r['name']:<28} {r['iou']:>5.2f} {str(r['drop']):>5} {r['seed']:>5} "
              + " ".join(f"{v:>5.1f}" for v in vals) + f" {r['mean']:>6.2f}")


def print_phase1(runs):
    print("\n" + "=" * 100)
    print("Phase 1 -- IoU threshold sweep (drop=False, seed=0)")
    print("=" * 100)
    p1 = [r for r in runs if not r["drop"] and r["seed"] == 0]
    p1.sort(key=lambda x: x["iou"])
    hdr = f"{'IoU':>5} " + " ".join(f"{s:>5}" for s in SHORT) + f" {'MEAN':>6}"
    print(hdr)
    print("-" * len(hdr))
    baselines_ttbbr = None
    baselines_amrod = None
    # Include AMROD-only reproduction (21.4) and TT-BBR-default (21.8) for reference
    for r in p1:
        vals = [r["per_corruption"][c] for c in CORR_ORDER]
        star = "  <--" if r["iou"] == max(p1, key=lambda x: x["mean"])["iou"] else ""
        print(f"{r['iou']:>5.2f} " + " ".join(f"{v:>5.1f}" for v in vals)
              + f" {r['mean']:>6.2f}{star}")


def print_phase3(runs):
    print("\n" + "=" * 100)
    print("Phase 3 -- Multi-seed variance")
    print("=" * 100)
    # Group by (iou, drop) that has more than 1 seed run.
    groups = {}
    for r in runs:
        key = (r["iou"], r["drop"])
        groups.setdefault(key, []).append(r)
    for (iou, drop), gr in sorted(groups.items()):
        if len(gr) < 2:
            continue
        seeds = sorted(r["seed"] for r in gr)
        means = [r["mean"] for r in gr]
        mn = statistics.mean(means)
        sd = statistics.stdev(means) if len(means) > 1 else 0.0
        print(f"iou={iou:.2f} drop={drop} seeds={seeds}: "
              f"mean = {mn:.2f} +/- {sd:.2f} (min {min(means):.2f}, max {max(means):.2f})")


def recommend(runs):
    print("\n" + "=" * 100)
    print("Recommendation")
    print("=" * 100)
    best = max(runs, key=lambda x: x["mean"])
    print(f"Best single run: {best['name']}  mean AP50 = {best['mean']:.2f}")
    print(f"    IoU={best['iou']:.2f}  drop={best['drop']}  seed={best['seed']}")
    print()
    print("For the paper, prefer the multi-seed mean +/- std at the winning config.")


def main():
    if not SWEEP_ROOT.is_dir():
        raise SystemExit(f"no sweep dir at {SWEEP_ROOT}")
    runs = load_all_runs()
    if not runs:
        print("[aggregate] no completed runs found")
        return
    print_all(runs)
    print_phase1(runs)
    print_phase3(runs)
    recommend(runs)


if __name__ == "__main__":
    main()
