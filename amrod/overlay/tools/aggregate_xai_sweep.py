"""Aggregate XAI-filter (Option 1) sweep results into a paper-ready table.

Reads all `/data/vgcmt/work_dirs/xai_short_sweep/*/stdout.log` files, parses
per-corruption AP50 and `AP50_ALL_mean`, produces:
  1. All runs sorted by mean.
  2. Phase 1 (technique comparison, TT-BBR off) as a line-per-method table.
  3. Phase 2 (technique comparison + TT-BBR) as a line-per-method table.
  4. Phase 3 (threshold sweep at BEST method) as a line-per-threshold table.
  5. Best-config recommendation vs the reproduced AMROD baseline (21.4) and
     the TT-BBR alone result (21.8).
"""
from __future__ import annotations

import re
from pathlib import Path

SWEEP_ROOT = Path("/data/vgcmt/work_dirs/xai_short_sweep")

# Reference points (from prior experiments):
AMROD_REPRO_MEAN = 21.4           # our D2 reproduction of AMROD (paper: 20.8)
TTBBR_ALONE_MEAN = 21.8           # TT-BBR added, iou=0.7 drop=False, random seed

CORR_ORDER = [
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
SHORT = ["Def", "Gls", "Mot", "Zoom", "Snow", "Frst", "Fog", "Brt",
         "Ctr", "Ela", "Pix", "JPG"]

# Name format: xai{method}_th{thresh}_{mode}_ttbbr{0|1}
RUN_RE = re.compile(
    r"^xai(?P<method>eigencam|featnorm|gradcam)"
    r"_th(?P<thresh>[\d.]+)"
    r"_(?P<mode>drop|reweight)"
    r"_ttbbr(?P<ttbbr>[01])$"
)


def parse_log(log_path: Path) -> dict | None:
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
    for cand in ("stdout.log", "log.txt"):
        p = run_dir / cand
        if p.is_file():
            with open(p) as f:
                if "AP50_ALL_mean:" in f.read():
                    return p
    return None


def load_all_runs() -> list[dict]:
    runs = []
    if not SWEEP_ROOT.is_dir():
        return runs
    for run_dir in sorted(SWEEP_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        log = find_run_log(run_dir)
        if log is None:
            print(f"[aggregate-xai] SKIP {run_dir.name} (incomplete)")
            continue
        data = parse_log(log)
        if data is None:
            continue
        runs.append(dict(
            name=run_dir.name,
            method=m.group("method"),
            thresh=float(m.group("thresh")),
            mode=m.group("mode"),
            ttbbr=bool(int(m.group("ttbbr"))),
            **data,
        ))
    return runs


def print_all_sorted(runs):
    print("\n=== All completed runs (sorted by mean AP50) ===")
    print(f"{'name':<40}  {'mean':>6}  {'Δvs21.4':>7}")
    for r in sorted(runs, key=lambda x: -x["mean"]):
        d = r["mean"] - AMROD_REPRO_MEAN
        print(f"{r['name']:<40}  {r['mean']:>6.2f}  {d:>+7.2f}")


def print_phase(runs, label, filter_fn, sort_key):
    subset = [r for r in runs if filter_fn(r)]
    if not subset:
        return
    print(f"\n=== {label} ===")
    print(f"{'method':<10} {'thresh':>7} {'mode':<9} {'ttbbr':<6} "
          f"{'mean':>6}  {'Δvs21.4':>7}  " + " ".join(f"{s:>4}" for s in SHORT))
    for r in sorted(subset, key=sort_key):
        d = r["mean"] - AMROD_REPRO_MEAN
        per = " ".join(f"{r['per_corruption'][c]:>4.1f}" for c in CORR_ORDER)
        print(f"{r['method']:<10} {r['thresh']:>7.2f} {r['mode']:<9} "
              f"{str(r['ttbbr']):<6} {r['mean']:>6.2f}  {d:>+7.2f}  {per}")


def print_best(runs):
    if not runs:
        return
    best = max(runs, key=lambda r: r["mean"])
    print("\n=== Best XAI config ===")
    print(f"  {best['name']}")
    print(f"  mean AP50 = {best['mean']:.2f}")
    print(f"  vs AMROD reproduction (21.4)  : {best['mean']-AMROD_REPRO_MEAN:+.2f}")
    print(f"  vs TT-BBR alone      (21.8)   : {best['mean']-TTBBR_ALONE_MEAN:+.2f}")


def main():
    runs = load_all_runs()
    if not runs:
        print("[aggregate-xai] no completed runs yet.")
        return

    print_all_sorted(runs)

    print_phase(runs,
                "Phase 1 -- XAI technique comparison (thresh=0.30, TT-BBR off)",
                lambda r: r["thresh"] == 0.30 and not r["ttbbr"],
                lambda r: r["method"])
    print_phase(runs,
                "Phase 2 -- XAI + TT-BBR composability (thresh=0.30, TT-BBR on)",
                lambda r: r["thresh"] == 0.30 and r["ttbbr"],
                lambda r: r["method"])
    print_phase(runs,
                "Phase 3 -- Threshold sweep (TT-BBR off)",
                lambda r: not r["ttbbr"],
                lambda r: (r["method"], r["thresh"]))

    print_best(runs)


if __name__ == "__main__":
    main()
