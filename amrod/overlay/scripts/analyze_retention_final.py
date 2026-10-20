from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/media/ilias/DATA/ilias/amrod_output/retention")
OUT = Path("results/retention_final")
OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["base", "lambda0", "lambda1", "taskcoco"]
DOMAIN_ORDER = [
    "fog_mtl",
    "motion_blur_mtl",
    "snow_mtl",
    "brightness_mtl",
    "defocus_blur_mtl",
]

METHOD_LABEL = {
    "base": "Base",
    "lambda0": "lambda=0",
    "lambda1": "lambda=1",
    "taskcoco": "Task-CoCo",
}

DOMAIN_LABEL = {
    "fog_mtl": "fog",
    "motion_blur_mtl": "motion",
    "snow_mtl": "snow",
    "brightness_mtl": "brightness",
    "defocus_blur_mtl": "defocus",
}


# ================================================================
# PAIRWISE
# ================================================================

pair_rows = []

for csv_path in sorted(ROOT.glob("retention_pair_*/retention.csv")):
    df = pd.read_csv(csv_path)

    probes = (
        df[df["kind"] == "probe_after_each"]
        .sort_values("state_idx")
        .reset_index(drop=True)
    )

    if len(probes) != 2:
        print(f"WARNING pairwise {csv_path}: expected 2 probes, got {len(probes)}")
        continue

    method = str(probes.iloc[0]["method"])
    before = probes.iloc[0]
    after = probes.iloc[1]

    challenger = str(after["adapted_dataset"])

    pair_rows.append({
        "method": method,
        "challenger": challenger,

        "fog_before_AP50": before["AP50"],
        "fog_after_AP50": after["AP50"],
        "delta_AP50": after["AP50"] - before["AP50"],

        "fog_before_mIoU": before["mIoU"],
        "fog_after_mIoU": after["mIoU"],
        "delta_mIoU": after["mIoU"] - before["mIoU"],
    })

pair = pd.DataFrame(pair_rows)

pair["method"] = pd.Categorical(
    pair["method"], METHOD_ORDER, ordered=True
)
pair["challenger"] = pd.Categorical(
    pair["challenger"],
    DOMAIN_ORDER[1:],
    ordered=True
)

pair = pair.sort_values(["method", "challenger"])
pair.to_csv(OUT / "pairwise_fog_interference.csv", index=False)


print("\n" + "=" * 110)
print("A. PAIRWISE: ISOLATED EFFECT ON FOG")
print("   delta = fog AFTER challenger - fog AFTER fog")
print("=" * 110)

show = pair.copy()
show["method"] = show["method"].astype(str).map(METHOD_LABEL)
show["challenger"] = show["challenger"].astype(str).map(DOMAIN_LABEL)

print(
    show[
        ["method", "challenger",
         "fog_before_AP50", "fog_after_AP50", "delta_AP50",
         "fog_before_mIoU", "fog_after_mIoU", "delta_mIoU"]
    ].round(3).to_string(index=False)
)


print("\nPAIRWISE ΔmIoU MATRIX")
piv = show.pivot(
    index="method",
    columns="challenger",
    values="delta_mIoU"
)
print(piv.round(3).to_string())


print("\nPAIRWISE ΔAP50 MATRIX")
piv = show.pivot(
    index="method",
    columns="challenger",
    values="delta_AP50"
)
print(piv.round(3).to_string())


# ================================================================
# LONGITUDINAL
# ================================================================

long_parts = []

for method in METHOD_ORDER:
    path = ROOT / f"retention_long_{method}_r3_s0" / "retention.csv"

    if not path.exists():
        print(f"WARNING missing {path}")
        continue

    df = pd.read_csv(path)
    df["method"] = method
    long_parts.append(df)

long = pd.concat(long_parts, ignore_index=True)
long.to_csv(OUT / "longitudinal_all.csv", index=False)


# ------------------------------------------------
# Dense frozen fog probes after every adaptation.
# ------------------------------------------------

fog = (
    long[
        (long["kind"] == "probe_after_each")
        & (long["probe_dataset"] == "fog_mtl")
    ]
    .copy()
    .sort_values(["method", "round", "position"])
)

fog["prev_AP50"] = fog.groupby(
    ["method", "round"]
)["AP50"].shift(1)

fog["prev_mIoU"] = fog.groupby(
    ["method", "round"]
)["mIoU"].shift(1)

fog["step_delta_AP50"] = fog["AP50"] - fog["prev_AP50"]
fog["step_delta_mIoU"] = fog["mIoU"] - fog["prev_mIoU"]


# Difference relative to state immediately after fog adaptation
fog["after_fog_AP50"] = fog.groupby(
    ["method", "round"]
)["AP50"].transform("first")

fog["after_fog_mIoU"] = fog.groupby(
    ["method", "round"]
)["mIoU"].transform("first")

fog["delta_from_fog_AP50"] = (
    fog["AP50"] - fog["after_fog_AP50"]
)

fog["delta_from_fog_mIoU"] = (
    fog["mIoU"] - fog["after_fog_mIoU"]
)

fog.to_csv(OUT / "longitudinal_fog_states.csv", index=False)


print("\n" + "=" * 110)
print("B. LONGITUDINAL: FROZEN FOG AFTER EVERY DOMAIN")
print("=" * 110)

for method in METHOD_ORDER:
    x = fog[fog["method"] == method]

    print(f"\n### {METHOD_LABEL[method]}")

    tab = x[
        [
            "round",
            "position",
            "adapted_dataset",
            "AP50",
            "mIoU",
            "step_delta_AP50",
            "step_delta_mIoU",
            "delta_from_fog_AP50",
            "delta_from_fog_mIoU",
        ]
    ].copy()

    tab["adapted_dataset"] = (
        tab["adapted_dataset"].map(DOMAIN_LABEL)
    )

    print(tab.round(3).to_string(index=False))


# ------------------------------------------------
# One row per method / round:
# fog state after each domain
# ------------------------------------------------

print("\n" + "=" * 110)
print("C. FOG mIoU TRAJECTORY BY ROUND")
print("=" * 110)

fog_traj = fog.pivot_table(
    index=["method", "round"],
    columns="adapted_dataset",
    values="mIoU",
    aggfunc="first",
).reset_index()

fog_traj["method"] = fog_traj["method"].map(METHOD_LABEL)

ordered_cols = [
    "method",
    "round",
    "fog_mtl",
    "motion_blur_mtl",
    "snow_mtl",
    "brightness_mtl",
    "defocus_blur_mtl",
]

fog_traj = fog_traj[ordered_cols]

fog_traj = fog_traj.rename(columns=DOMAIN_LABEL)

fog_traj["fog_to_end"] = (
    fog_traj["defocus"] - fog_traj["fog"]
)

print(fog_traj.round(3).to_string(index=False))
fog_traj.to_csv(OUT / "fog_trajectory_by_round.csv", index=False)


# ------------------------------------------------
# Transition-specific effects in longitudinal stream
# ------------------------------------------------

transition = fog[fog["position"] > 1].copy()

transition["transition"] = (
    transition["adapted_dataset"].map({
        "motion_blur_mtl": "fog_state -> motion",
        "snow_mtl": "motion_state -> snow",
        "brightness_mtl": "snow_state -> brightness",
        "defocus_blur_mtl": "brightness_state -> defocus",
    })
)

print("\n" + "=" * 110)
print("D. LONGITUDINAL INCREMENTAL TRANSITION EFFECTS ON FOG mIoU")
print("=" * 110)

tr = transition[
    ["method", "round", "transition", "step_delta_AP50", "step_delta_mIoU"]
].copy()

tr["method"] = tr["method"].map(METHOD_LABEL)

print(tr.round(3).to_string(index=False))

tr.to_csv(OUT / "longitudinal_transition_effects.csv", index=False)


# Mean effect of each transition across three rounds
mean_tr = (
    transition
    .groupby(["method", "adapted_dataset"], as_index=False)
    .agg(
        mean_delta_AP50=("step_delta_AP50", "mean"),
        mean_delta_mIoU=("step_delta_mIoU", "mean"),
        min_delta_mIoU=("step_delta_mIoU", "min"),
        max_delta_mIoU=("step_delta_mIoU", "max"),
    )
)

print("\n" + "=" * 110)
print("E. MEAN LONGITUDINAL TRANSITION EFFECT ACROSS R1-R3")
print("=" * 110)

tmp = mean_tr.copy()
tmp["method"] = tmp["method"].map(METHOD_LABEL)
tmp["adapted_dataset"] = tmp["adapted_dataset"].map(DOMAIN_LABEL)

print(tmp.round(3).to_string(index=False))
mean_tr.to_csv(OUT / "mean_transition_effects.csv", index=False)


# ================================================================
# ROUND-END FULL RETENTION SNAPSHOT
# ================================================================

snap_rows = []

for (method, rnd), grp in long.groupby(["method", "round"]):

    # Fog: same model state, after final domain, from dense probe.
    fog_row = grp[
        (grp["kind"] == "probe_after_each")
        & (grp["position"] == 5)
        & (grp["probe_dataset"] == "fog_mtl")
    ]

    if len(fog_row) == 1:
        r = fog_row.iloc[0]
        snap_rows.append({
            "method": method,
            "round": rnd,
            "dataset": "fog_mtl",
            "AP50": r["AP50"],
            "mIoU": r["mIoU"],
        })

    # Remaining four frozen probes at exactly same state.
    rr = grp[grp["kind"] == "probe_round_end"]

    for _, r in rr.iterrows():
        snap_rows.append({
            "method": method,
            "round": rnd,
            "dataset": r["probe_dataset"],
            "AP50": r["AP50"],
            "mIoU": r["mIoU"],
        })


snap = pd.DataFrame(snap_rows)
snap.to_csv(OUT / "round_end_full_snapshot.csv", index=False)

round_ret = (
    snap.groupby(["method", "round"], as_index=False)
    .agg(
        frozen_mean_AP50=("AP50", "mean"),
        frozen_mean_mIoU=("mIoU", "mean"),
    )
)

print("\n" + "=" * 110)
print("F. FROZEN 5-DOMAIN RETENTION AT END OF EACH ROUND")
print("=" * 110)

rr = round_ret.copy()
rr["method"] = rr["method"].map(METHOD_LABEL)

print(rr.round(3).to_string(index=False))
round_ret.to_csv(OUT / "round_end_mean_retention.csv", index=False)


# ================================================================
# R3 END-STATE DOMAIN SNAPSHOT
# ================================================================

r3 = snap[snap["round"] == 3].copy()

print("\n" + "=" * 110)
print("G. R3 FINAL FROZEN STATE -- ALL DOMAINS")
print("=" * 110)

r3show = r3.copy()
r3show["method"] = r3show["method"].map(METHOD_LABEL)
r3show["dataset"] = r3show["dataset"].map(DOMAIN_LABEL)

print(
    r3show[
        ["method", "dataset", "AP50", "mIoU"]
    ].round(3).to_string(index=False)
)

r3.to_csv(OUT / "r3_final_domain_snapshot.csv", index=False)


# ================================================================
# CUMULATIVE FOG RETENTION R1 -> R3
# ================================================================

rows = []

for method in METHOD_ORDER:
    x = fog[fog["method"] == method]

    start = x[
        (x["round"] == 1)
        & (x["position"] == 1)
    ].iloc[0]

    r1_end = x[
        (x["round"] == 1)
        & (x["position"] == 5)
    ].iloc[0]

    r3_end = x[
        (x["round"] == 3)
        & (x["position"] == 5)
    ].iloc[0]

    rows.append({
        "method": METHOD_LABEL[method],

        "R1_after_fog_AP50": start["AP50"],
        "R1_end_AP50": r1_end["AP50"],
        "R3_end_AP50": r3_end["AP50"],

        "R1_after_fog_mIoU": start["mIoU"],
        "R1_end_mIoU": r1_end["mIoU"],
        "R3_end_mIoU": r3_end["mIoU"],

        "within_R1_delta_mIoU":
            r1_end["mIoU"] - start["mIoU"],

        "cumulative_R1start_to_R3end_mIoU":
            r3_end["mIoU"] - start["mIoU"],
    })

cum = pd.DataFrame(rows)

print("\n" + "=" * 110)
print("H. CUMULATIVE FOG RETENTION")
print("=" * 110)
print(cum.round(3).to_string(index=False))

cum.to_csv(OUT / "cumulative_fog_retention.csv", index=False)


print("\nSaved everything to:")
print(OUT.resolve())
