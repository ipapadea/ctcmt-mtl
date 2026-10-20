from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results/taskcoco_drift_analysis")
PER_EVAL = ROOT / "per_eval.csv"
OUT = ROOT / "drift"
OUT.mkdir(parents=True, exist_ok=True)


RUNS = {
    "Base": "newsrc_base_cscLT_s0",
    "lambda=0": "newsrc_ctcr_rblock_cscLT_s0",
    "lambda=1": "newsrc_ctcr_cscLT_s0",
    "Task-CoCo": "newsrc_taskcoco_cscLT_s0",
}

DOMAIN_ORDER = [
    "fog_mtl",
    "motion_blur_mtl",
    "snow_mtl",
    "brightness_mtl",
    "defocus_blur_mtl",
]


df = pd.read_csv(PER_EVAL)

# ---------------------------------------------------------------------
# Keep only the four CSC-LT runs.
# ---------------------------------------------------------------------
parts = []

for method, run in RUNS.items():
    x = df[df["run"] == run].copy()

    if len(x) != 50:
        raise RuntimeError(
            f"{method}: expected 50 evaluations for {run}, found {len(x)}"
        )

    x = x.sort_values("idx").reset_index(drop=True)
    x["method"] = method

    # Five domains per round.
    x["round"] = x["idx"] // 5 + 1
    x["domain_pos"] = x["idx"] % 5

    expected = [DOMAIN_ORDER[i] for i in x["domain_pos"]]

    if list(x["dataset"]) != expected:
        raise RuntimeError(
            f"{method}: stream order does not match expected CSC-LT cycle."
        )

    parts.append(x)

all_df = pd.concat(parts, ignore_index=True)

all_df.to_csv(OUT / "all_eval_points.csv", index=False)


# ---------------------------------------------------------------------
# 1. Round means
# ---------------------------------------------------------------------
round_summary = (
    all_df
    .groupby(["method", "round"], as_index=False)
    .agg(
        AP50=("AP50", "mean"),
        mIoU=("mIoU", "mean"),
    )
)

round_summary.to_csv(OUT / "round_summary.csv", index=False)

print("\n" + "=" * 90)
print("ROUND-WISE MEANS")
print("=" * 90)

for method in RUNS:
    x = round_summary[round_summary["method"] == method]
    print(f"\n{method}")
    print(x.to_string(index=False))


# ---------------------------------------------------------------------
# 2. Domain means over all ten visits
# ---------------------------------------------------------------------
domain_summary = (
    all_df
    .groupby(["method", "dataset"], as_index=False)
    .agg(
        AP50=("AP50", "mean"),
        mIoU=("mIoU", "mean"),
        AP50_std=("AP50", "std"),
        mIoU_std=("mIoU", "std"),
    )
)

domain_summary["dataset"] = pd.Categorical(
    domain_summary["dataset"],
    categories=DOMAIN_ORDER,
    ordered=True,
)

domain_summary = domain_summary.sort_values(["method", "dataset"])
domain_summary.to_csv(OUT / "domain_summary.csv", index=False)

print("\n" + "=" * 90)
print("DOMAIN MEANS ACROSS 10 ROUNDS")
print("=" * 90)
print(domain_summary.to_string(index=False))


# ---------------------------------------------------------------------
# 3. R1 -> R10 change
# ---------------------------------------------------------------------
rows = []

for method in RUNS:
    x = round_summary[round_summary["method"] == method].set_index("round")

    r1 = x.loc[1]
    r10 = x.loc[10]

    rows.append({
        "method": method,
        "R1_AP50": r1.AP50,
        "R10_AP50": r10.AP50,
        "delta_AP50": r10.AP50 - r1.AP50,
        "R1_mIoU": r1.mIoU,
        "R10_mIoU": r10.mIoU,
        "delta_mIoU": r10.mIoU - r1.mIoU,
    })

r1_r10 = pd.DataFrame(rows)
r1_r10.to_csv(OUT / "r1_r10.csv", index=False)

print("\n" + "=" * 90)
print("R1 -> R10")
print("=" * 90)
print(r1_r10.to_string(index=False))


# ---------------------------------------------------------------------
# 4. Peak mIoU and subsequent degradation
# ---------------------------------------------------------------------
rows = []

for method in RUNS:
    x = round_summary[round_summary["method"] == method].copy()

    peak_idx = x["mIoU"].idxmax()
    peak = x.loc[peak_idx]

    r10 = x[x["round"] == 10].iloc[0]

    rows.append({
        "method": method,
        "peak_round": int(peak["round"]),
        "peak_mIoU": peak["mIoU"],
        "R10_mIoU": r10["mIoU"],
        "drop_peak_to_R10": r10["mIoU"] - peak["mIoU"],
    })

peak_df = pd.DataFrame(rows)
peak_df.to_csv(OUT / "miou_peak_drift.csv", index=False)

print("\n" + "=" * 90)
print("SEGMENTATION PEAK -> R10 DRIFT")
print("=" * 90)
print(peak_df.to_string(index=False))


# ---------------------------------------------------------------------
# 5. Per-domain R1 -> R10 change
# ---------------------------------------------------------------------
rows = []

for method in RUNS:
    for domain in DOMAIN_ORDER:
        x = all_df[
            (all_df["method"] == method)
            & (all_df["dataset"] == domain)
        ].sort_values("round")

        first = x.iloc[0]
        last = x.iloc[-1]

        rows.append({
            "method": method,
            "dataset": domain,
            "R1_AP50": first.AP50,
            "R10_AP50": last.AP50,
            "delta_AP50": last.AP50 - first.AP50,
            "R1_mIoU": first.mIoU,
            "R10_mIoU": last.mIoU,
            "delta_mIoU": last.mIoU - first.mIoU,
        })

domain_drift = pd.DataFrame(rows)
domain_drift.to_csv(OUT / "domain_r1_r10.csv", index=False)

print("\n" + "=" * 90)
print("PER-DOMAIN R1 -> R10")
print("=" * 90)
print(domain_drift.to_string(index=False))


# ---------------------------------------------------------------------
# 6. Correlation between detection improvement and segmentation change.
#    This is diagnostic only, not causal.
# ---------------------------------------------------------------------
rows = []

for method in RUNS:
    x = round_summary[round_summary["method"] == method].sort_values("round")

    corr_level = x["AP50"].corr(x["mIoU"])

    d_ap = x["AP50"].diff().dropna()
    d_miou = x["mIoU"].diff().dropna()

    corr_delta = d_ap.corr(d_miou)

    rows.append({
        "method": method,
        "corr_AP50_mIoU_levels": corr_level,
        "corr_round_delta_AP50_delta_mIoU": corr_delta,
    })

corr_df = pd.DataFrame(rows)
corr_df.to_csv(OUT / "correlations.csv", index=False)

print("\n" + "=" * 90)
print("AP50 / mIoU CORRELATIONS")
print("=" * 90)
print(corr_df.to_string(index=False))


# ---------------------------------------------------------------------
# 7. Relative to Base at each round
# ---------------------------------------------------------------------
base = (
    round_summary[round_summary["method"] == "Base"]
    [["round", "AP50", "mIoU"]]
    .rename(columns={
        "AP50": "Base_AP50",
        "mIoU": "Base_mIoU",
    })
)

relative = round_summary.merge(base, on="round")
relative["delta_vs_base_AP50"] = relative["AP50"] - relative["Base_AP50"]
relative["delta_vs_base_mIoU"] = relative["mIoU"] - relative["Base_mIoU"]

relative.to_csv(OUT / "delta_vs_base_by_round.csv", index=False)


# ---------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------
plt.figure(figsize=(9, 5))

for method in RUNS:
    x = round_summary[round_summary["method"] == method]
    plt.plot(x["round"], x["AP50"], marker="o", label=method)

plt.xlabel("Round")
plt.ylabel("AP50")
plt.xticks(range(1, 11))
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "csc_ap50_by_round.png", dpi=200)
plt.close()


plt.figure(figsize=(9, 5))

for method in RUNS:
    x = round_summary[round_summary["method"] == method]
    plt.plot(x["round"], x["mIoU"], marker="o", label=method)

plt.xlabel("Round")
plt.ylabel("mIoU")
plt.xticks(range(1, 11))
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "csc_miou_by_round.png", dpi=200)
plt.close()


# Per-domain mIoU trajectories.
for domain in DOMAIN_ORDER:
    plt.figure(figsize=(9, 5))

    for method in RUNS:
        x = all_df[
            (all_df["method"] == method)
            & (all_df["dataset"] == domain)
        ].sort_values("round")

        plt.plot(
            x["round"],
            x["mIoU"],
            marker="o",
            label=method,
        )

    plt.xlabel("Round")
    plt.ylabel("mIoU")
    plt.title(domain)
    plt.xticks(range(1, 11))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    safe_name = domain.replace("_mtl", "")
    plt.savefig(
        OUT / f"csc_{safe_name}_miou_by_round.png",
        dpi=200,
    )
    plt.close()


print("\nSaved analysis to:")
print(OUT.resolve())
