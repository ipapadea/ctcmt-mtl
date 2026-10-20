"""Aggregate per-corruption D2 results.json files into a single summary.json
in the same schema as tools/aggregate_cityscapes_c.py (mmdet side)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AMROD_CORRUPTIONS = [
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost",
    "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

PAPER_TABLE2 = {
    "defocus_blur": 6.8, "glass_blur": 8.1, "motion_blur": 8.0, "zoom_blur": 1.5,
    "snow": 0.2, "frost": 6.8, "fog": 34.6, "brightness": 30.7, "contrast": 3.0,
    "elastic_transform": 50.2, "pixelate": 17.6, "jpeg_compression": 13.5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work-root", required=True)
    args = ap.parse_args()

    root = Path(args.work_root)
    per_corr = {}
    missing = []
    for c in AMROD_CORRUPTIONS:
        p = root / c / "results.json"
        if not p.is_file():
            missing.append(c)
            continue
        with open(p) as f:
            per_corr[c] = json.load(f)

    summary = {
        "run_name": args.run_name,
        "checkpoint": args.ckpt,
        "pipeline": "detectron2",
        "per_corruption": {c: {"mAP": r["mAP"], "mAP_50": r["mAP_50"]}
                           for c, r in per_corr.items()},
        "missing_corruptions": missing,
    }
    if per_corr:
        summary["mean_mAP"] = sum(r["mAP"] for r in per_corr.values()) / len(per_corr)
        summary["mean_mAP_50"] = sum(r["mAP_50"] for r in per_corr.values()) / len(per_corr)

    out_path = root / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    hdr = f"{'corruption':<18} {'paper':>7} {'ours':>7} {'delta':>7}"
    print("\n" + "=" * len(hdr))
    print(f"D2 pipeline eval  --  run={args.run_name}")
    print(hdr); print("-" * len(hdr))
    for c in AMROD_CORRUPTIONS:
        p = PAPER_TABLE2[c]
        if c in per_corr:
            v = per_corr[c]["mAP_50"] * 100
            print(f"{c:<18} {p:>7.1f} {v:>7.1f} {v - p:>+7.1f}")
        else:
            print(f"{c:<18} {p:>7.1f} {'MISS':>7} {'':>7}")
    if per_corr:
        mean = summary["mean_mAP_50"] * 100
        pmean = sum(PAPER_TABLE2.values()) / len(PAPER_TABLE2)
        print(f"{'MEAN':<18} {pmean:>7.1f} {mean:>7.1f} {mean - pmean:>+7.1f}")
    print("=" * len(hdr))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
