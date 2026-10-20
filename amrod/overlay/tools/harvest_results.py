#!/usr/bin/env python3
"""Harvest all CTTA experiment results into results/registry.json.

Scans known log locations (detectron2 d2 + TriLiteNet), extracts metrics,
and writes a structured JSON registry. Run at any time — idempotent.

Usage:
    python3 tools/harvest_results.py            # update registry
    python3 tools/harvest_results.py --print    # pretty-print current registry
    python3 tools/harvest_results.py --print --filter acdc   # filter by tag
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REGISTRY_FILE = RESULTS_DIR / "registry.json"

# ---------------------------------------------------------------------------
# D2 log scanner
# ---------------------------------------------------------------------------

D2_CORRUPTIONS = [
    "defocus_blur","glass_blur","motion_blur","zoom_blur",
    "snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression",
]
ACDC_WEATHERS = ["acdc_fog","acdc_night","acdc_rain","acdc_snow"]

ACDC_MTL_WEATHERS = [
    "acdc_fog_mtl",
    "acdc_night_mtl",
    "acdc_rain_mtl",
    "acdc_snow_mtl",
]

CS_C_LT_CORRUPTIONS = [
    "fog",
    "motion_blur",
    "snow",
    "brightness",
    "defocus_blur",
]

def _parse_d2_log(
    logfile: Path,
    expected_cycle: Optional[List[str]] = None,
    expected_rounds: int = 1,
) -> Optional[Dict]:
    """Parse a detectron2 CTTA log.

    When an expected protocol is provided, only the latest complete
    occurrence of that exact protocol is returned.

    This is important because Detectron2 log.txt files may contain
    multiple independent runs appended over time. Repeated dataset
    names must therefore not automatically be interpreted as CTTA
    rounds.

    Examples:
      ACDC short:
        1 × [fog, night, rain, snow]

      ACDC long-term:
        10 × [fog, night, rain, snow]

      Cityscapes-C short:
        1 × 12 corruptions

      Cityscapes-C long-term:
        10 × [fog, motion_blur, snow, brightness, defocus_blur]
    """
    if not logfile.is_file():
        return None

    text = logfile.read_text(errors="replace")
    if "CTTA_" not in text and "Evaluation results for" not in text:
        return None

    # Collect all evaluation results in chronological log order.
    occurrences = []

    for m in re.finditer(
        r"Evaluation results for (\S+) in csv format.*?copypaste: ([0-9.,]+)",
        text,
        re.DOTALL,
    ):
        ds, cp = m.group(1), m.group(2)

        vals = [float(v) for v in cp.split(",")]

        if len(vals) >= 2:
            occurrences.append(
                (
                    ds,
                    vals[0] / 100.0,
                    vals[1] / 100.0,
                )
            )

    if not occurrences:
        return None

    # ------------------------------------------------------------------
    # Benchmark-aware parsing.
    #
    # Search backwards for the latest COMPLETE exact protocol.
    # This prevents old runs appended to log.txt from being interpreted
    # as additional continual-adaptation rounds.
    # ------------------------------------------------------------------
    if expected_cycle:
        allowed = set(expected_cycle)

        # Remove unrelated evaluations that may exist in the same log.
        filtered = [
            x for x in occurrences
            if x[0] in allowed
        ]

        expected_names = list(expected_cycle) * expected_rounds
        n_expected = len(expected_names)

        if len(filtered) < n_expected:
            return None

        selected = None

        # Find the latest exact complete protocol.
        for start in range(len(filtered) - n_expected, -1, -1):
            window = filtered[start:start + n_expected]
            names = [x[0] for x in window]

            if names == expected_names:
                selected = window
                break

        if selected is None:
            return None

        # Single-round experiment.
        if expected_rounds == 1:
            return {
                ds: {
                    "AP": ap,
                    "AP50": ap50,
                }
                for ds, ap, ap50 in selected
            }

        # Multi-round experiment.
        per_repeat = []

        cycle_len = len(expected_cycle)

        for r in range(expected_rounds):
            block = selected[
                r * cycle_len:(r + 1) * cycle_len
            ]

            per_repeat.append({
                ds: {
                    "AP": ap,
                    "AP50": ap50,
                }
                for ds, ap, ap50 in block
            })

        return {
            "__per_repeat__": per_repeat
        }

    # ------------------------------------------------------------------
    # Generic fallback for benchmarks for which no explicit protocol
    # has been defined yet.
    # ------------------------------------------------------------------
    ds_names = [o[0] for o in occurrences]

    is_multi = len(ds_names) > len(set(ds_names))

    if not is_multi:
        return {
            ds: {
                "AP": ap,
                "AP50": ap50,
            }
            for ds, ap, ap50 in occurrences
        }

    first_ds = ds_names[0]

    per_repeat = []
    current = {}

    for ds, ap, ap50 in occurrences:
        if ds == first_ds and current:
            per_repeat.append(current)
            current = {}

        current[ds] = {
            "AP": ap,
            "AP50": ap50,
        }

    if current:
        per_repeat.append(current)

    return {
        "__per_repeat__": per_repeat
    }


def _d2_summary(results: Dict) -> Dict:
    """Compute summary from a d2 result dict (single or multi-round)."""
    if "__per_repeat__" in results:
        per_repeat = results["__per_repeat__"]
        # Mean AP50 across all rounds × all datasets (overall mean).
        all_ap50 = [v["AP50"] for r in per_repeat for v in r.values()]
        mean_all = round(sum(all_ap50) / len(all_ap50), 4) if all_ap50 else None
        # Per-round mean AP50 for R1/R4/R7/R10 reporting.
        round_means = []
        for r in per_repeat:
            ap50s = [v["AP50"] for v in r.values()]
            round_means.append(round(sum(ap50s) / len(ap50s), 4) if ap50s else None)
        return {
            "mean_AP50": mean_all,
            "n_rounds": len(per_repeat),
            "n_datasets_per_round": len(per_repeat[0]) if per_repeat else 0,
            "round_mean_AP50": round_means,  # index 0 = R1, index 3 = R4, etc.
        }
    # Single-round.
    ap50s = [v["AP50"] for v in results.values() if "AP50" in v]
    return {"mean_AP50": round(sum(ap50s) / len(ap50s), 4) if ap50s else None,
            "n_datasets": len(results)}


# ---------------------------------------------------------------------------
# TriLiteNet log scanner
# ---------------------------------------------------------------------------

def _parse_trilit_summary(summary_json: Path) -> Optional[Dict]:
    """Parse a TriLiteNet adapt_ctta summary.json."""
    if not summary_json.is_file():
        return None
    try:
        d = json.loads(summary_json.read_text())
    except Exception:
        return None
    out = {
        "global_miou": d.get("global", {}).get("miou"),
        "global_AP50": d.get("global", {}).get("AP50"),
        "per_weather": {},
        "per_repeat": [],
    }
    for w, v in d.get("per_weather", {}).items():
        out["per_weather"][w] = {"miou": v.get("miou"), "AP50": v.get("AP50")}
    for r in d.get("per_repeat", []):
        out["per_repeat"].append({
            "miou": r.get("miou"),
            "det": {w: v.get("AP50") for w, v in r.get("det", {}).items()},
        })
    return out


# ---------------------------------------------------------------------------
# Known experiment registry
# ---------------------------------------------------------------------------

D2_OUTPUT_ROOT = Path("/data/ilias/panoptic_fpn/output/ctta_acdc")
CS_C_ROOT = Path("/data/ilias/panoptic_fpn/output/ctta_cs_c")
TRILIT_CTTA_ROOT = Path("/home/ilias/TriLiteNet/runs/ctta")

D2_EXPERIMENTS = [
    # tag, output_subdir, benchmark, description
    # subdir is relative to D2_OUTPUT_ROOT for all entries (CS-C also lands in ctta_acdc/)
    ("amrod_acdc",          "amrod",               "ACDC",    "AMROD, MkRCNN src, Cityscapes→ACDC"),
    ("amrod_official_acdc", "amrod_official",      "ACDC",    "AMROD, official src, Cityscapes→ACDC"),
    ("ctcmt_det_acdc",      "ctcmt_det",           "ACDC",    "CT-CMT-Det, PFN src, Cityscapes→ACDC"),
    ("ctcmt_seg_acdc",      "ctcmt_seg",           "ACDC",    "CT-CMT-Seg, PFN src, Cityscapes→ACDC"),
    ("ctcmt_mtl_acdc",      "ctcmt_mtl",           "ACDC",    "CT-CMT-MTL v4b, PFN src, Cityscapes→ACDC"),
    ("amrod_x10",           "amrod_x10",           "ACDC-10", "AMROD long-term 10-round"),
    ("ctcmt_det_x10",       "ctcmt_det_x10",       "ACDC-10", "CT-CMT-Det long-term 10-round"),
    ("ctcmt_v4b_x10",       "ctcmt_v4b_x10",       "ACDC-10", "CT-CMT-MTL v4b long-term 10-round"),
    ("amrod_shift",         "amrod_shift",         "SHIFT",   "AMROD, Cityscapes→SHIFT"),
    ("ctcmt_det_shift",     "ctcmt_det_shift",     "SHIFT",   "CT-CMT-Det, Cityscapes→SHIFT"),
    # Foggy Cityscapes
    ("source_foggy",        "source_only_foggy",   "Foggy",   "Source-only, Cityscapes→FoggyCS"),
    ("amrod_foggy",         "amrod_foggy_cs",      "Foggy",   "AMROD, Cityscapes→FoggyCS"),
    ("amrod_v2_foggy",      "amrod_v2_foggy",      "Foggy",   "AMROD+V2, Cityscapes→FoggyCS"),
    ("ctcmt_v2_foggy_mtl",  "ctcmt_v2_foggy_mtl",  "Foggy",   "CT-CMT-MTL v2, Cityscapes→FoggyCS"),
    ("ctcmt_v4b_foggy_mtl", "ctcmt_v4b_foggy_mtl", "Foggy",   "CT-CMT-MTL v4b (best), Cityscapes→FoggyCS"),
    ("ctcmt_det_mr_foggy",  "ctcmt_det_mr_foggy",  "Foggy",   "CT-CMT-Det MkRCNN src, Cityscapes→FoggyCS"),
    # Cityscapes-C (output lands in ctta_acdc/<track_name>/ due to OUT_ROOT override)
    ("amrod_cs_c",          "amrod_cs_c",          "CS-C",    "AMROD, MkRCNN-ours src, CS→CS-C"),
    ("amrod_official_cs_c", "amrod_official_cs_c", "CS-C",    "AMROD, official src, CS→CS-C"),
    ("ctcmt_det_cs_c",      "ctcmt_det_cs_c",      "CS-C",    "CT-CMT-Det, PFN src, CS→CS-C"),
    ("ctcmt_det_mr_cs_c",   "ctcmt_det_mr_cs_c",   "CS-C",    "CT-CMT-Det, MkRCNN src, CS→CS-C"),
    ("ctcmt_seg_cs_c",      "ctcmt_seg_cs_c",      "CS-C",    "CT-CMT-Seg, PFN src, CS→CS-C"),
    ("ctcmt_mtl_cs_c",      "ctcmt_mtl_cs_c",      "CS-C",    "CT-CMT-MTL v4b, PFN src, CS→CS-C"),
    ("cotta_cs_c",          "cotta_cs_c",          "CS-C",    "CoTTA-Seg, PFN src, CS→CS-C"),
    # CS-C long-term (Table 3: 5 corruptions × 10 rounds)
    ("amrod_cs_c_lt",          "amrod_cs_c_lt_x10",          "CS-C-LT", "AMROD, MkRCNN src, CS→CS-C 5-corr long-term"),
    ("amrod_official_cs_c_lt", "amrod_official_cs_c_lt_x10", "CS-C-LT", "AMROD, official src, CS→CS-C 5-corr long-term"),
    ("ctcmt_det_mr_cs_c_lt",   "ctcmt_det_mr_cs_c_lt_x10",   "CS-C-LT", "CT-CMT-Det MkRCNN, CS→CS-C 5-corr long-term"),
    ("ctcmt_seg_cs_c_lt",      "ctcmt_seg_cs_c_lt_x10",      "CS-C-LT", "CT-CMT-Seg PFN, CS→CS-C 5-corr long-term"),
    ("ctcmt_mtl_cs_c_lt",      "ctcmt_mtl_cs_c_lt_x10",      "CS-C-LT", "CT-CMT-MTL v4b, CS→CS-C 5-corr long-term"),
]

TRILIT_EXPERIMENTS = [
    # tag, run_name, benchmark, description
    # --- MTL source ---
    ("trilit_source_acdc",      "cityscapes_mtl_base_native__source__x1",     "ACDC",    "TriLiteNet MTL, source-only"),
    ("trilit_amrod_acdc",       "cityscapes_mtl_base_native__amrod_mtl__x1",  "ACDC",    "TriLiteNet MTL, AMROD-MTL adapter"),
    ("trilit_ctcmt_acdc",       "cityscapes_mtl_base_native__ctcmt__x1",      "ACDC",    "TriLiteNet MTL, CT-CMT adapter"),
    ("trilit_ctcmt_x10",        "cityscapes_mtl_base_native__ctcmt__x10",     "ACDC-10", "TriLiteNet MTL, CT-CMT long-term"),
    ("trilit_ctcmt_det_acdc",   "cityscapes_mtl_base_native__ctcmt_det__x1",  "ACDC",    "TriLiteNet MTL, CT-CMT det-only"),
    ("trilit_ctcmt_seg_acdc",   "cityscapes_mtl_base_native__ctcmt_seg__x1",  "ACDC",    "TriLiteNet MTL, CT-CMT seg-only"),
    ("trilit_ctcmt_mtl_acdc",   "cityscapes_mtl_base_native__ctcmt_mtl__x1",  "ACDC",    "TriLiteNet MTL, CT-CMT MTL-only"),
    # --- Det-only source ---
    ("trilit_det_source_acdc",  "cityscapes_det_base_native__source__x1",     "ACDC",    "TriLiteNet det-only, source-only"),
    ("trilit_det_amrod_acdc",   "cityscapes_det_base_native__amrod__x1",      "ACDC",    "TriLiteNet det-only, AMROD adapter"),
    ("trilit_det_ctcmt_acdc",   "cityscapes_det_base_native__ctcmt__x1",      "ACDC",    "TriLiteNet det-only, CT-CMT adapter"),
    ("trilit_det_amrod_x10",    "cityscapes_det_base_native__amrod__x10",     "ACDC-10", "TriLiteNet det-only, AMROD long-term"),
    # --- Seg-only source ---
    ("trilit_seg_source_acdc",  "cityscapes_seg_base_native__source__x1",     "ACDC",    "TriLiteNet seg-only, source-only"),
    ("trilit_seg_cotta_acdc",   "cityscapes_seg_base_native__cotta__x1",      "ACDC",    "TriLiteNet seg-only, CoTTA adapter"),
    ("trilit_seg_ctcmt_acdc",   "cityscapes_seg_base_native__ctcmt__x1",      "ACDC",    "TriLiteNet seg-only, CT-CMT adapter"),
    ("trilit_seg_cotta_x10",    "cityscapes_seg_base_native__cotta__x10",     "ACDC-10", "TriLiteNet seg-only, CoTTA long-term"),
]


# ---------------------------------------------------------------------------
# Main harvester
# ---------------------------------------------------------------------------

def _d2_protocol(tag: str, benchmark: str):
    """Return (expected_dataset_cycle, expected_rounds)."""

    # Cityscapes -> ACDC long-term (AMROD Table 5)
    if benchmark == "ACDC-10":
        return ACDC_WEATHERS, 10

    # Cityscapes -> Cityscapes-C short-term (AMROD Table 2)
    if benchmark == "CS-C":
        return D2_CORRUPTIONS, 1

    # Cityscapes -> Cityscapes-C long-term (AMROD Table 3)
    if benchmark == "CS-C-LT":
        return CS_C_LT_CORRUPTIONS, 10

    # Short ACDC experiments.
    if benchmark == "ACDC":
        # Panoptic-FPN / MTL configs use the *_mtl registrations.
        if tag in {
            "ctcmt_det_acdc",
            "ctcmt_seg_acdc",
            "ctcmt_mtl_acdc",
        }:
            return ACDC_MTL_WEATHERS, 1

        return ACDC_WEATHERS, 1

    # SHIFT / Foggy / other future benchmarks:
    # keep the generic parser for now.
    return None, 1


def harvest():
    entries = {}
    invalid_tags = set()

    # D2 experiments
    for tag, subdir, bench, desc in D2_EXPERIMENTS:
        logf = D2_OUTPUT_ROOT / subdir / "log.txt"
        cycle, rounds = _d2_protocol(tag, bench)
        parsed = _parse_d2_log(logf, expected_cycle=cycle, expected_rounds=rounds)
        if parsed:
            entries[tag] = {
                "tag": tag,
                "benchmark": bench,
                "description": desc,
                "framework": "detectron2",
                "log": str(logf),
                "summary": _d2_summary(parsed),
                "per_dataset": parsed,
                "harvested_at": datetime.utcnow().isoformat(),
            }
        elif logf.is_file() and cycle is not None:
            # Log exists, explicit protocol defined, but no valid complete run found.
            invalid_tags.add(tag)

    # TriLiteNet experiments
    for tag, run_name, bench, desc in TRILIT_EXPERIMENTS:
        summary_f = TRILIT_CTTA_ROOT / run_name / "summary.json"
        parsed = _parse_trilit_summary(summary_f)
        if parsed:
            entries[tag] = {
                "tag": tag,
                "benchmark": bench,
                "description": desc,
                "framework": "trilitenet",
                "summary_json": str(summary_f),
                "metrics": parsed,
                "harvested_at": datetime.utcnow().isoformat(),
            }

    return entries, invalid_tags


def pretty_print(registry: Dict, filter_str: Optional[str] = None) -> None:
    rows = sorted(registry.values(), key=lambda x: (x["benchmark"], x["tag"]))
    if filter_str:
        rows = [r for r in rows if filter_str.lower() in json.dumps(r).lower()]

    bench = None
    for r in rows:
        if r["benchmark"] != bench:
            bench = r["benchmark"]
            print(f"\n=== {bench} ===")
        tag = r["tag"]
        desc = r["description"]
        s = r.get("summary") or r.get("metrics", {})
        mean_ap50 = s.get("mean_AP50") or s.get("global_AP50")
        miou = s.get("global_miou")
        metrics = ""
        if mean_ap50 is not None:
            metrics += f"  AP50={mean_ap50*100:.1f}%"
        # Show R1 and R10 for long-term experiments.
        rmeans = s.get("round_mean_AP50", [])
        if len(rmeans) >= 2:
            r1 = rmeans[0]*100 if rmeans[0] is not None else float("nan")
            r10 = rmeans[-1]*100 if rmeans[-1] is not None else float("nan")
            metrics += f"  (R1={r1:.1f}% R{len(rmeans)}={r10:.1f}%)"
        if miou is not None:
            metrics += f"  mIoU={miou*100:.1f}%"
        print(f"  {tag:<35} {desc:<45}{metrics}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="Print registry to stdout")
    ap.add_argument("--filter", default=None, help="Filter by substring")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REGISTRY_FILE.read_text()) if REGISTRY_FILE.is_file() else {}

    harvested, invalid_tags = harvest()

    for tag in sorted(invalid_tags):
        if tag in existing:
            print(
                f"[harvest] Removing stale entry '{tag}': "
                "existing log does not contain a complete valid protocol."
            )
            existing.pop(tag)

    existing.update(harvested)

    REGISTRY_FILE.write_text(json.dumps(existing, indent=2))
    print(f"[harvest] {len(harvested)} entries harvested → {REGISTRY_FILE}")
    print(f"[harvest] Registry total: {len(existing)} entries")

    if args.print:
        pretty_print(existing, args.filter)


if __name__ == "__main__":
    main()
