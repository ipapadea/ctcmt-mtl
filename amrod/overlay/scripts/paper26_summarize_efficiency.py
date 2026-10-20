#!/usr/bin/env python3

import glob
import json
import os
import re
import statistics

ROOT = os.environ.get(
    "PAPER26_EFFICIENCY_ROOT",
    "/media/ilias/DATA/ilias/amrod_output/paper26_efficiency",
)

# Original four-method table: always required.
ORDER = [
    ("Base", "base"),
    ("+CTCR", "ctcr"),
    ("Task-CoCo", "taskcoco"),
    ("Ours-lambda1", "ours"),
]

# SOTA baselines: included once all 3 repeats exist (never partially).
SOTA = [
    ("AMROD", "amrod"),
    ("CoTTA", "cotta"),
    ("TENT", "tent"),
]

TASKS = {
    "Base": "Detection + Segmentation",
    "+CTCR": "Detection + Segmentation",
    "Task-CoCo": "Detection + Segmentation",
    "Ours-lambda1": "Detection + Segmentation",
    "AMROD": "Detection",
    "CoTTA": "Segmentation",
    "TENT": "Segmentation",
}

EXPECTED_DATASET = {
    "Base": "defocus_blur_mtl",
    "+CTCR": "defocus_blur_mtl",
    "Task-CoCo": "defocus_blur_mtl",
    "Ours-lambda1": "defocus_blur_mtl",
    "AMROD": "defocus_blur",
    "CoTTA": "defocus_blur_semseg",
    "TENT": "defocus_blur_semseg",
}

# Only the SOTA runs record these (older JSONs predate the fields).
EXPECTED_META = {
    "AMROD": "AMROD",
    "CoTTA": "CoTTA_SemSeg",
    "TENT": "TENT_SemSeg",
}
EXPECTED_CONFIG = {
    "AMROD": "amrod_newsrc_csc.yaml",
    "CoTTA": "cotta_newsrc_csc.yaml",
    "TENT": "tent_newsrc_csc.yaml",
}
CKPT_SUFFIX = "panoptic_fpn_R50_cityscapes_segw1/model_final.pth"

MULTI_TASK = {label for label, _ in ORDER}


def load(tag):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, f"{tag}_r*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def mu_sd(xs):
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def corruption_of(dataset):
    return re.sub(r"_(mtl|semseg)$", "", dataset)


all_rows = {}
methods = []

# ------------------------------------------------------------------
# Load + same-method consistency
# ------------------------------------------------------------------

for label, tag in ORDER + SOTA:
    rows = load(tag)

    if label in MULTI_TASK:
        if len(rows) != 3:
            raise RuntimeError(
                f"{label}: expected 3 repeats, found {len(rows)}"
            )
    else:
        if len(rows) == 0:
            print(f"[note] {label}: no runs found, omitted from the table")
            continue
        if len(rows) != 3:
            raise RuntimeError(
                f"{label}: partial run, expected 3 repeats, found {len(rows)}"
            )

    all_rows[label] = rows
    methods.append((label, tag))

    reference = rows[0]

    for i, r in enumerate(rows[1:], start=2):
        for key in [
            "config_file",
            "dataset",
            "gpu",
            "measured_images",
            "warmup_images",
            "seed",
            "resolved_config",
            "task_scope",
            "meta_architecture",
            "model_weights",
        ]:
            if r.get(key) != reference.get(key):
                raise RuntimeError(
                    f"{label}: repeat {i} differs in {key}: "
                    f"{r.get(key)!r} != {reference.get(key)!r}"
                )

    ref = rows[0]

    if ref["dataset"] != EXPECTED_DATASET[label]:
        raise RuntimeError(
            f"{label}: dataset {ref['dataset']!r} != "
            f"{EXPECTED_DATASET[label]!r}"
        )

    scope = ref.get("task_scope")
    if scope and scope != TASKS[label]:
        raise RuntimeError(
            f"{label}: task_scope {scope!r} != {TASKS[label]!r}"
        )

    if label in EXPECTED_META:
        if ref.get("meta_architecture") != EXPECTED_META[label]:
            raise RuntimeError(
                f"{label}: meta_architecture {ref.get('meta_architecture')!r}"
                f" != {EXPECTED_META[label]!r}"
            )
        if not str(ref.get("model_weights", "")).endswith(CKPT_SUFFIX):
            raise RuntimeError(
                f"{label}: not the canonical checkpoint: "
                f"{ref.get('model_weights')!r}"
            )
        if not str(ref["config_file"]).endswith(EXPECTED_CONFIG[label]):
            raise RuntimeError(
                f"{label}: unexpected config {ref['config_file']!r}"
            )


# ------------------------------------------------------------------
# Cross-method benchmark fairness
# ------------------------------------------------------------------

global_ref = all_rows["Base"][0]

for label, _ in methods:
    r = all_rows[label][0]

    for key in [
        "gpu",
        "measured_images",
        "warmup_images",
        "seed",
    ]:
        if r[key] != global_ref[key]:
            raise RuntimeError(
                f"{label}: benchmark setup differs from Base in {key}: "
                f"{r[key]!r} != {global_ref[key]!r}"
            )

    # Same corruption; the dataset *view* (det / seg / mtl) follows the task.
    if corruption_of(r["dataset"]) != corruption_of(global_ref["dataset"]):
        raise RuntimeError(
            f"{label}: corruption differs from Base: "
            f"{r['dataset']!r} vs {global_ref['dataset']!r}"
        )


# ------------------------------------------------------------------
# Exact method provenance
# ------------------------------------------------------------------

base_cfg = all_rows["Base"][0]["resolved_config"]

assert abs(float(base_cfg["ctcr_weight"])) < 1e-8, base_cfg
assert abs(float(base_cfg["ctcl_weight"])) < 1e-8, base_cfg
assert base_cfg["ctpv"] is False, base_cfg


ctcr_cfg = all_rows["+CTCR"][0]["resolved_config"]

assert abs(float(ctcr_cfg["ctcr_weight"]) - 0.3) < 1e-8, ctcr_cfg
assert abs(float(ctcr_cfg["ctcl_weight"])) < 1e-8, ctcr_cfg
assert ctcr_cfg["ctpv"] is False, ctcr_cfg


task_cfg = all_rows["Task-CoCo"][0]["resolved_config"]

assert task_cfg["conflict_mode"] == "task_coco", task_cfg
assert abs(float(task_cfg["ctcr_weight"]) - 0.3) < 1e-8, task_cfg
assert abs(float(task_cfg["ctcl_weight"])) < 1e-8, task_cfg
assert task_cfg["ctpv"] is False, task_cfg


ours_cfg = all_rows["Ours-lambda1"][0]["resolved_config"]

assert ours_cfg["conflict_mode"] == "resnet_route", ours_cfg
assert abs(float(ours_cfg["route_lambda"]) - 1.0) < 1e-8, ours_cfg
assert ours_cfg["adaptive_routing"] is False, ours_cfg
assert abs(float(ours_cfg["ctcr_weight"]) - 0.3) < 1e-8, ours_cfg
assert abs(float(ours_cfg["ctcl_weight"])) < 1e-8, ours_cfg
assert ours_cfg["ctpv"] is False, ours_cfg


# ------------------------------------------------------------------
# Baseline references
# ------------------------------------------------------------------

base_latency = statistics.mean(
    [x["mean_ms_per_image"] for x in all_rows["Base"]]
)

base_memory = max(
    x["peak_allocated_mib"] for x in all_rows["Base"]
)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print()
print("=" * 140)
print("PAPER26 EFFICIENCY — 3 REPEATS")
print("=" * 140)

print(
    f"{'Method':<15}"
    f"{'Task(s)':<27}"
    f"{'Latency ms/img':>20}"
    f"{'FPS':>16}"
    f"{'p95 ms':>17}"
    f"{'Peak MiB':>14}"
    f"{'Overhead':>13}"
    f"{'Trainable M':>14}"
)

summary = []

for label, tag in methods:
    rows = all_rows[label]

    lat = [x["mean_ms_per_image"] for x in rows]
    fps = [x["fps_from_mean"] for x in rows]
    p95 = [x["p95_ms_per_image"] for x in rows]

    lm, ls = mu_sd(lat)
    fm_raw, fs = mu_sd(fps)
    pm, ps = mu_sd(p95)

    # FPS consistent with the reported mean latency.
    fm = 1000.0 / lm

    peak = max(x["peak_allocated_mib"] for x in rows)

    # Base-relative overhead is only meaningful inside the multi-task family;
    # detection-only / segmentation-only workloads are not comparable to Base.
    if label in MULTI_TASK:
        overhead = 100.0 * (lm / base_latency - 1.0)
        mem_overhead = peak - base_memory
        overhead_str = f"{overhead:+12.1f}%"
    else:
        overhead = None
        mem_overhead = None
        overhead_str = f"{'-':>13}"

    trainable_m = rows[0]["trainable_params"] / 1e6
    total_m = rows[0]["total_params"] / 1e6
    student_m = rows[0]["student_params"] / 1e6

    print(
        f"{label:<15}"
        f"{TASKS[label]:<27}"
        f"{lm:9.2f} ± {ls:<8.2f}"
        f"{fm:7.2f} ± {fs:<6.2f}"
        f"{pm:8.2f} ± {ps:<6.2f}"
        f"{peak:14.1f}"
        f"{overhead_str}"
        f"{trainable_m:14.2f}"
    )

    summary.append({
        "method": label,
        "task_scope": TASKS[label],
        "dataset": rows[0]["dataset"],
        "latency_mean_ms": lm,
        "latency_std_ms": ls,
        "fps_mean": fm,
        "fps_std": fs,
        "p95_mean_ms": pm,
        "p95_std_ms": ps,
        "peak_mib": peak,
        "latency_overhead_vs_base_pct": overhead,
        "memory_overhead_vs_base_mib": mem_overhead,
        "trainable_m": trainable_m,
        "total_m": total_m,
        "student_m": student_m,
    })

print()
print(
    "Overhead is relative to Base within the multi-task family only; "
    "'-' = single-task workload, not comparable."
)


# ------------------------------------------------------------------
# Diagnostic: lambda=1 routing execution overhead
# ------------------------------------------------------------------

print()
print("Diagnostic: lambda=1 execution overhead relative to plain +CTCR")

ctcr = next(x for x in summary if x["method"] == "+CTCR")
ours = next(x for x in summary if x["method"] == "Ours-lambda1")

route_overhead = 100.0 * (
    ours["latency_mean_ms"] / ctcr["latency_mean_ms"] - 1.0
)

print(
    f"Ours-lambda1 vs +CTCR latency overhead: "
    f"{route_overhead:+.2f}%"
)


# ------------------------------------------------------------------
# Save derived summary
# ------------------------------------------------------------------

# Written under a new name: the original four-method efficiency_summary.json
# is a completed canonical artifact and is left untouched.
out = os.path.join(ROOT, "efficiency_summary_all.json")

with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print()
print("Saved:", out)
