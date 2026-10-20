#!/usr/bin/env python3

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "historical_finaltables"
INPUT = RESULTS / "per_seed.csv"

METRICS = [
    "bbox_ap50",
    "miou",
    "r1_bbox_ap50",
    "r10_bbox_ap50",
    "r1_miou",
    "r10_miou",
]

DATASET_FILES = {
    "acdc-lt": "ACDC_LT.md",
    "cs-c": "CITYSCAPES_C_SHORT.md",
    "cs-c-lt": "CITYSCAPES_C_LT.md",
}

DATASET_TITLES = {
    "acdc-lt": "Cityscapes → ACDC Long-Term",
    "cs-c": "Cityscapes → Cityscapes-C Short",
    "cs-c-lt": "Cityscapes → Cityscapes-C Long-Term",
}

SOURCE_ORDER = {
    "Mask R-CNN R50-FPN": 0,
    "Semantic-FPN R50": 1,
    "Panoptic-FPN R50": 2,
}


def ffloat(x):
    return None if x == "" else float(x)


with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 27:
    raise RuntimeError(f"Expected 27 experiments, found {len(rows)}")

for r in rows:
    if r["status"] != "VALID":
        raise RuntimeError(f"Non-valid row: {r}")
    if int(r["actual_evals"]) != int(r["expected_evals"]):
        raise RuntimeError(
            f"Evaluation-count mismatch: {r['method']} "
            f"{r['dataset']} seed={r['seed']}"
        )
    for m in METRICS:
        r[m] = ffloat(r[m])

groups = defaultdict(list)

for r in rows:
    key = (
        r["dataset"],
        r["source_group"],
        r["source_checkpoint"],
        r["method"],
        r["runner_alias"],
        r["config"],
    )
    groups[key].append(r)

for key, vals in groups.items():
    seeds = sorted(int(x["seed"]) for x in vals)
    if seeds != [0, 42, 123]:
        raise RuntimeError(f"{key}: expected seeds [0,42,123], got {seeds}")


def stat(vals, field):
    x = [r[field] for r in vals if r[field] is not None]
    if not x:
        return None, None
    if len(x) != 3:
        raise RuntimeError(f"{field}: expected 3 values, found {len(x)}")
    return statistics.mean(x), statistics.stdev(x)


summary = []

for key, vals in sorted(groups.items()):
    dataset, source, checkpoint, method, alias, config = key

    row = {
        "dataset": dataset,
        "source_group": source,
        "source_checkpoint": checkpoint,
        "method": method,
        "runner_alias": alias,
        "config": config,
        "n_seeds": 3,
    }

    for metric in METRICS:
        mean, std = stat(vals, metric)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std

    summary.append(row)


summary_fields = [
    "dataset",
    "source_group",
    "source_checkpoint",
    "method",
    "runner_alias",
    "config",
    "n_seeds",
]

for metric in METRICS:
    summary_fields += [f"{metric}_mean", f"{metric}_std"]


with (RESULTS / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader()

    for r in summary:
        out = dict(r)
        for metric in METRICS:
            for suffix in ("mean", "std"):
                k = f"{metric}_{suffix}"
                if out[k] is not None:
                    out[k] = f"{out[k]:.6f}"
                else:
                    out[k] = ""
        w.writerow(out)


def fmt(mean, std):
    if mean is None:
        return "—"
    return f"{mean:.3f} ± {std:.3f}"


for dataset, filename in DATASET_FILES.items():
    ds_rows = [r for r in summary if r["dataset"] == dataset]

    protocol = next(
        r["protocol"] for r in rows if r["dataset"] == dataset
    )

    lines = [
        f"# {DATASET_TITLES[dataset]}",
        "",
        f"**Protocol:** `{protocol}`  ",
        "**Canonical seeds:** `0, 42, 123`  ",
        "**Uncertainty:** sample standard deviation (`n-1`, `n=3`).",
        "",
        "> Fair-comparison rule: methods should be compared as method "
        "ablations only inside the same source-checkpoint section. "
        "Values across different source sections are architectural/reference "
        "comparisons, not controlled same-source comparisons.",
        "",
    ]

    sources = sorted(
        {r["source_group"] for r in ds_rows},
        key=lambda x: SOURCE_ORDER.get(x, 999),
    )

    for source in sources:
        src_rows = [r for r in ds_rows if r["source_group"] == source]
        checkpoint = src_rows[0]["source_checkpoint"]

        lines += [
            f"## Source: {source}",
            "",
            f"Checkpoint: `{checkpoint}`",
            "",
            "| Method | n | AP50 | mIoU | R1 AP50 | R10 AP50 | "
            "R1 mIoU | R10 mIoU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for r in src_rows:
            lines.append(
                f"| {r['method']} | {r['n_seeds']} | "
                f"{fmt(r['bbox_ap50_mean'], r['bbox_ap50_std'])} | "
                f"{fmt(r['miou_mean'], r['miou_std'])} | "
                f"{fmt(r['r1_bbox_ap50_mean'], r['r1_bbox_ap50_std'])} | "
                f"{fmt(r['r10_bbox_ap50_mean'], r['r10_bbox_ap50_std'])} | "
                f"{fmt(r['r1_miou_mean'], r['r1_miou_std'])} | "
                f"{fmt(r['r10_miou_mean'], r['r10_miou_std'])} |"
            )

        lines += [
            "",
            "Config:",
            f"`{src_rows[0]['config']}`",
            "",
        ]

    lines += [
        "## Raw per-seed values",
        "",
        "The exact seed-level values used to compute these tables are stored "
        "in [`per_seed.csv`](per_seed.csv).",
        "",
    ]

    (RESULTS / filename).write_text("\n".join(lines))


readme = """# Historical FINAL_TABLES benchmark archive

This directory freezes the canonical historical benchmark results collected
before implementing the EXTENSION changes.

## Code provenance

- Branch: `ft-historical-benchmarks`
- Commit: `4dc18a84e75b1b2802f28a9b1c057cc056060ac2`
- Canonical seeds: `0`, `42`, `123`
- Canonical experiments: `27 / 27`
- Detection metric: bbox `AP50`
- Segmentation metric: `mIoU`
- Standard deviation: sample standard deviation (`n-1`)

## Benchmark tables

- [`ACDC_LT.md`](ACDC_LT.md)
- [`CITYSCAPES_C_SHORT.md`](CITYSCAPES_C_SHORT.md)
- [`CITYSCAPES_C_LT.md`](CITYSCAPES_C_LT.md)

Machine-readable results:

- [`per_seed.csv`](per_seed.csv): one row per canonical experiment.
- [`summary.csv`](summary.csv): mean ± sample-std components.

## Fair comparison rule

The exact source checkpoint is part of the experimental condition.

The canonical historical methods use three different source checkpoints:

- `CTCMT_Det [historical FT]`:
  `mask_rcnn_R50_cityscapes/model_final.pth`
- `CTCMT_Seg [historical FT]`:
  `semantic_R50_cityscapes/model_final.pth`
- `CTCMT-MTL+V2 [historical FT]`:
  `panoptic_fpn_R50_cityscapes/model_final.pth`

Therefore, numbers from different source groups may be shown together as
cross-architecture references, but they must not be interpreted as controlled
method-only comparisons.

Future Panoptic-FPN ablations and EXTENSION variants should be compared inside
the Panoptic-FPN source group using the same source checkpoint.

## Validation

All 27 canonical runs have the expected evaluation counts:

- ACDC-LT: 40 evaluations/run.
- Cityscapes-C short: 12 evaluations/run.
- Cityscapes-C LT: 50 evaluations/run.

Seed 0 was additionally checked across all 9 logs with zero tracebacks.

On Cronus, the initial attempts for the following Cityscapes-C runs failed
before evaluation because the Cityscapes COCO annotation JSON was not reachable
at the expected filesystem path:

- seed 123: Det CS-C, Det CS-C-LT, MTL CS-C, MTL CS-C-LT.
- seed 42: Det CS-C, Det CS-C-LT.

The dataset filesystem path was repaired and only those affected experiments
were rerun with the same method configuration, source weights, benchmark,
and seed. The successful reruns are the canonical values recorded here.

## Raw-output locations

Historical raw outputs were produced under:

`/data/ilias/panoptic_fpn/output/ft_historical/`

Seed 0 was run on `gpu1`; seeds 42 and 123 were run on `Cronus`.
"""

(RESULTS / "README.md").write_text(readme)

print("VALIDATION PASSED")
print("27 / 27 canonical experiments")
print("3 / 3 seeds per method × benchmark")
print()
for p in [
    RESULTS / "README.md",
    RESULTS / "ACDC_LT.md",
    RESULTS / "CITYSCAPES_C_SHORT.md",
    RESULTS / "CITYSCAPES_C_LT.md",
    RESULTS / "per_seed.csv",
    RESULTS / "summary.csv",
]:
    print(p.relative_to(ROOT))
