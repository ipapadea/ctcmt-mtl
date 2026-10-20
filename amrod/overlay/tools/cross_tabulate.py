"""Final 2x2 (and 2x3) cross-framework comparison table:

                   cityscapes_c        cityscapes_c_amrod    cityscapes_c_amrod
                   (mmdet gen)         (AMROD gen)           (AMROD gen)
                   mmdet pipeline      mmdet pipeline        D2 pipeline
  ------------------------------------------------------------------------------
  AMROD ckpt         [historical]        [done]              [needs D2 run]
  Ours COCO ckpt     [historical]        [done]              [needs D2 run]

Reads six summary.json files and prints the aggregated comparison plus
paper-15.1 sanity check.
"""
from __future__ import annotations

import json
from pathlib import Path

AMROD_CORRUPTIONS = [
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost",
    "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
SHORT = ["Def", "Gls", "Mot", "Zoom", "Snow", "Frst", "Fog", "Brt", "Ctr", "Ela", "Pix", "JPG"]
PAPER = {"defocus_blur": 6.8, "glass_blur": 8.1, "motion_blur": 8.0, "zoom_blur": 1.5,
         "snow": 0.2, "frost": 6.8, "fog": 34.6, "brightness": 30.7, "contrast": 3.0,
         "elastic_transform": 50.2, "pixelate": 17.6, "jpeg_compression": 13.5}

CELLS = [
    ("mmdet_ourcorr_amrod",   "/data/vgcmt/work_dirs/cityscapes_c_source_only_amrod/summary.json"),
    ("mmdet_ourcorr_ours",    "/data/vgcmt/work_dirs/cityscapes_c_source_only_ours_coco/summary.json"),
    ("mmdet_amrodcorr_amrod", "/data/vgcmt/work_dirs/cityscapes_c_source_only_amrod_amrodcorr/summary.json"),
    ("mmdet_amrodcorr_ours",  "/data/vgcmt/work_dirs/cityscapes_c_source_only_ours_coco_amrodcorr/summary.json"),
    ("d2_amrodcorr_amrod",    "/data/vgcmt/work_dirs/amrod_d2_pipeline/amrod_ckpt/summary.json"),
    ("d2_amrodcorr_ours",     "/data/vgcmt/work_dirs/amrod_d2_pipeline/ours_ckpt/summary.json"),
]


def load(path):
    p = Path(path)
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def row(name, s):
    if s is None:
        return f"{name:<28} " + " ".join(f"{'--':>5}" for _ in SHORT) + f" {'--':>7}"
    per = s["per_corruption"]
    vals = [per.get(c, {}).get("mAP_50", 0.0) * 100 for c in AMROD_CORRUPTIONS]
    mean = sum(vals) / len(vals)
    return f"{name:<28} " + " ".join(f"{v:>5.1f}" for v in vals) + f" {mean:>7.2f}"


def main():
    data = {name: load(p) for name, p in CELLS}

    hdr = f"{'':<28} " + " ".join(f"{s:>5}" for s in SHORT) + f" {'MEAN':>7}"
    sep = "-" * len(hdr)
    print(sep)
    print("D2-vs-mmdet cross-framework Cityscapes-C source-only (mAP@50 %)")
    print(sep)
    print(hdr)
    print(sep)
    print(f"{'PAPER Table 2 Source':<28} " + " ".join(f"{PAPER[c]:>5.1f}" for c in AMROD_CORRUPTIONS)
          + f" {sum(PAPER.values())/12:>7.2f}")
    print(sep)
    print(row("mmdet + our-corr   + AMROD",  data["mmdet_ourcorr_amrod"]))
    print(row("mmdet + AMROD-corr + AMROD", data["mmdet_amrodcorr_amrod"]))
    print(row("D2    + AMROD-corr + AMROD", data["d2_amrodcorr_amrod"]))
    print(sep)
    print(row("mmdet + our-corr   + OURS",   data["mmdet_ourcorr_ours"]))
    print(row("mmdet + AMROD-corr + OURS",  data["mmdet_amrodcorr_ours"]))
    print(row("D2    + AMROD-corr + OURS",  data["d2_amrodcorr_ours"]))
    print(sep)

    def mean(s):
        return s.get("mean_mAP_50", 0.0) * 100 if s else None

    print()
    print("2x2 on cityscapes_c_amrod (SAME images, both pipelines) -- mean mAP@50:")
    print(f"{'ckpt':<10} {'D2':>10} {'mmdet':>10} {'delta D2-mmdet':>18}")
    for ckpt in ("amrod", "ours"):
        d2 = mean(data[f"d2_amrodcorr_{ckpt}"])
        mm = mean(data[f"mmdet_amrodcorr_{ckpt}"])
        d2s = f"{d2:.2f}" if d2 is not None else "--"
        mms = f"{mm:.2f}" if mm is not None else "--"
        d = f"{d2-mm:+.2f}" if (d2 is not None and mm is not None) else "--"
        print(f"{ckpt:<10} {d2s:>10} {mms:>10} {d:>18}")

    print()
    print("Ckpt-effect within each pipeline (mean mAP@50):")
    print(f"{'pipeline':<10} {'AMROD':>10} {'OURS':>10} {'delta OURS-AMROD':>20}")
    for pipe, key in (("mmdet", "mmdet_amrodcorr"), ("D2", "d2_amrodcorr")):
        a = mean(data[f"{key}_amrod"])
        o = mean(data[f"{key}_ours"])
        a_s = f"{a:.2f}" if a is not None else "--"
        o_s = f"{o:.2f}" if o is not None else "--"
        d = f"{o-a:+.2f}" if (a is not None and o is not None) else "--"
        print(f"{pipe:<10} {a_s:>10} {o_s:>10} {d:>20}")

    paper_mean = sum(PAPER.values()) / 12
    d2_amrod = mean(data["d2_amrodcorr_amrod"])
    print()
    if d2_amrod is not None:
        print(f"Paper Table 2 reproduction: D2+AMROD-corr+AMROD-ckpt = {d2_amrod:.2f}  "
              f"vs paper {paper_mean:.2f}  ({d2_amrod - paper_mean:+.2f})")
    else:
        print(f"Paper Table 2 reproduction: PENDING (no d2_amrodcorr_amrod yet)")


if __name__ == "__main__":
    main()
