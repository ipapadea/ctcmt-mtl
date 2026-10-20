#!/usr/bin/env python3
"""Pre-flight check for the S1..S5 negative-transfer screening batch.

Fully resolves each arm's config (following _BASE_ chains) and diffs it against
E13a, so a run can only start once the ONLY difference is the intended factor.
Also asserts the shared protocol invariants the batch depends on.

  python scripts/verify_screening_configs.py
"""
import os
import sys

from detectron2.config import get_cfg

BASE = "detectron2/configs/Cityscapes/ctcmt_e13a_thrmax080.yaml"
CFGDIR = "detectron2/configs/Cityscapes"

_GRAD = {"SOLVER.CTCMT_CONFLICT_MODE", "SOLVER.CTCMT_GRAD_DIAG", "OUTPUT_DIR"}

ARMS = {
    "e16_protectedgrad": (f"{CFGDIR}/ctcmt_e16_protectedgrad.yaml", _GRAD),
    "e17_cagrad": (f"{CFGDIR}/ctcmt_e17_cagrad.yaml", _GRAD),
    "e18_harddecouple": (f"{CFGDIR}/ctcmt_e18_harddecouple.yaml", _GRAD),
    "e19_frozentrunk": (f"{CFGDIR}/ctcmt_e19_frozentrunk.yaml", {
        "SOLVER.CTCMT_FREEZE_SHARED_TRUNK", "OUTPUT_DIR",
    }),
    "e20_dynweight": (f"{CFGDIR}/ctcmt_e20_dynweight.yaml", _GRAD),
    "e22_seghead_only": (f"{CFGDIR}/ctcmt_e22_seghead_only.yaml", _GRAD),
    "e23_detseg_nocross": (f"{CFGDIR}/ctcmt_e23_detseg_nocross.yaml", {
        "SOLVER.CTCMT_CTCL_ENABLED", "SOLVER.CTCMT_WEIGHT_CTCR", "OUTPUT_DIR",
    }),
}

# Everything that must be identical across the batch: the source checkpoint,
# the dynamic-threshold / gate machinery, restoration, CT-CR mode, augmentation.
INVARIANTS = {
    "SOLVER.THRESHOLD_MAX": 0.80,
    "SOLVER.CTCMT_CTPV_ENABLED": False,
    "SOLVER.CTCMT_DET_ONLY": False,
    "SOLVER.CTCMT_SEG_ONLY": False,
    "SOLVER.CTCMT_CTCR_MODE": "soft_seg_global",
    "SOLVER.CTCMT_CTCR_WEIGHT_FLOOR": 0.2,
    "SOLVER.CTCMT_CLASS_BALANCED_CE": True,
    "SOLVER.CTCMT_SEG_LOSS_SCALE_PRESERVE": True,
    "SOLVER.CTCMT_FISHER_RESTORE": False,
    "SOLVER.CTCMT_PROTO_ANCHOR": False,
    "SOLVER.CTCMT_GRAD_DIAG_EVERY": 50,
    "SEED": 0,
}

CYCLE = ("fog", "motion_blur", "snow", "brightness", "defocus_blur")
STREAM = tuple(f"{c}_mtl" for _ in range(10) for c in CYCLE)


def flatten(node, prefix=""):
    out = {}
    for k, v in node.items():
        key = f"{prefix}{k}"
        if hasattr(v, "items"):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load(path):
    cfg = get_cfg()
    cfg.merge_from_file(path)
    # Exactly the CLI overrides run_mixed_lt_local.sh appends.
    cfg.merge_from_list(["SEED", "0", "DATASETS.TEST", repr(list(STREAM))])
    return cfg


base = flatten(load(BASE))
failures = []

print(f"BASE = {BASE}")
print(f"  MODEL.WEIGHTS            = {base['MODEL.WEIGHTS']}")
print(f"  MODEL.META_ARCHITECTURE  = {base['MODEL.META_ARCHITECTURE']}")
print(f"  SOLVER.THRESHOLD_MAX     = {base['SOLVER.THRESHOLD_MAX']}")
print(f"  stream = ({' -> '.join(CYCLE)}) x 10 = {len(STREAM)} evaluations")
assert len(STREAM) == 50, "the long-term stream must contain exactly 50 evaluations"
print()

for name, (path, allowed) in ARMS.items():
    cfg = flatten(load(path))
    diff = {k for k in set(base) | set(cfg) if base.get(k, "<missing>") != cfg.get(k, "<missing>")}
    unexpected = diff - allowed
    missing = allowed - diff - {"OUTPUT_DIR"}

    print(f"--- {name}  ({path})")
    for k in sorted(diff):
        mark = "  " if k in allowed else "!!"
        print(f"  {mark} {k}: {base.get(k, '<missing>')!r} -> {cfg.get(k, '<missing>')!r}")
    if len(cfg["DATASETS.TEST"]) != 50:
        failures.append(f"{name}: stream has {len(cfg['DATASETS.TEST'])} evaluations, expected 50")
    if unexpected:
        failures.append(f"{name}: unintended diff vs E13a: {sorted(unexpected)}")
    if missing:
        failures.append(f"{name}: expected factor not applied: {sorted(missing)}")

    for k, want in INVARIANTS.items():
        got = cfg.get(k, "<missing>")
        if got != want:
            failures.append(f"{name}: {k} = {got!r}, expected {want!r}")
    # Strong aug must stay on everywhere; e23 is the only arm allowed to drop
    # the cross-task losses, so CT-CR's weight is checked per-arm.
    if cfg.get("SOLVER.CTCMT_STRONG_AUG_STUDENT") is not True:
        failures.append(f"{name}: strong augmentation was disabled")
    if name != "e23_detseg_nocross":
        if cfg.get("SOLVER.CTCMT_WEIGHT_CTCR") != base["SOLVER.CTCMT_WEIGHT_CTCR"]:
            failures.append(f"{name}: CT-CR weight changed")
        if cfg.get("SOLVER.CTCMT_CTCL_ENABLED") != base["SOLVER.CTCMT_CTCL_ENABLED"]:
            failures.append(f"{name}: CT-CL enable flag changed")
    if cfg["MODEL.WEIGHTS"] != base["MODEL.WEIGHTS"]:
        failures.append(f"{name}: source checkpoint changed")
    if cfg["OUTPUT_DIR"] == base["OUTPUT_DIR"]:
        failures.append(f"{name}: OUTPUT_DIR collides with E13a")
    print()

outs = [flatten(load(p))["OUTPUT_DIR"] for p, _ in ARMS.values()]
if len(set(outs)) != len(outs):
    failures.append(f"duplicate OUTPUT_DIRs: {outs}")

# --- ACDC arms are one factor from E11, not E13a. The 0.80 threshold ceiling
# was a Cityscapes-C-specific fix and is deliberately NOT part of the ACDC
# recipe, so these are checked against their own base.
ACDC_BASE = f"{CFGDIR}/ctcmt_e11_bothsc_ctcrD.yaml"
ACDC_ARMS = {
    "e24_seghead_only_acdc": (f"{CFGDIR}/ctcmt_e24_seghead_only_acdc.yaml", {
        "SOLVER.CTCMT_CONFLICT_MODE", "SOLVER.CTCMT_GRAD_DIAG", "OUTPUT_DIR",
    }),
    "e25_detonly_acdc": (f"{CFGDIR}/ctcmt_e25_detonly_acdc.yaml", {
        "SOLVER.CTCMT_DET_ONLY", "OUTPUT_DIR",
    }),
}
acdc_base = flatten(load(ACDC_BASE))
print(f"ACDC BASE = {ACDC_BASE}  (THRESHOLD_MAX={acdc_base['SOLVER.THRESHOLD_MAX']})")
if acdc_base["SOLVER.THRESHOLD_MAX"] != 0.90:
    failures.append("ACDC base must keep THRESHOLD_MAX 0.90")
for name, (path, allowed) in ACDC_ARMS.items():
    cfg = flatten(load(path))
    diff = {k for k in set(acdc_base) | set(cfg)
            if acdc_base.get(k, "<missing>") != cfg.get(k, "<missing>")}
    print(f"--- {name}  ({path})")
    for k in sorted(diff):
        mark = "  " if k in allowed else "!!"
        print(f"  {mark} {k}: {acdc_base.get(k, '<missing>')!r} -> {cfg.get(k, '<missing>')!r}")
    if diff - allowed:
        failures.append(f"{name}: unintended diff vs E11: {sorted(diff - allowed)}")
    if cfg["MODEL.WEIGHTS"] != acdc_base["MODEL.WEIGHTS"]:
        failures.append(f"{name}: source checkpoint changed")
    if cfg["SOLVER.THRESHOLD_MAX"] != 0.90:
        failures.append(f"{name}: THRESHOLD_MAX must stay 0.90 on ACDC")
    print()

# --- Options O1-O3: one factor each from their S6 parent, on both protocols.
OPTION_ARMS = {
    "e26_partial025":           (f"{CFGDIR}/ctcmt_e26_partial025.yaml",
                                 f"{CFGDIR}/ctcmt_e22_seghead_only.yaml",
                                 {"SOLVER.CTCMT_AUX_TRUNK_LAMBDA", "OUTPUT_DIR"}),
    "e27_s6_entropy":           (f"{CFGDIR}/ctcmt_e27_s6_entropy.yaml",
                                 f"{CFGDIR}/ctcmt_e22_seghead_only.yaml",
                                 {"SOLVER.CTCMT_ENTROPY_WEIGHTED_CE", "OUTPUT_DIR"}),
    "e28_adaptive_routing":     (f"{CFGDIR}/ctcmt_e28_adaptive_routing.yaml",
                                 f"{CFGDIR}/ctcmt_e26_partial025.yaml",
                                 {"SOLVER.CTCMT_ADAPTIVE_ROUTING", "OUTPUT_DIR"}),
    "e26_partial025_acdc":      (f"{CFGDIR}/ctcmt_e26_partial025_acdc.yaml",
                                 f"{CFGDIR}/ctcmt_e24_seghead_only_acdc.yaml",
                                 {"SOLVER.CTCMT_AUX_TRUNK_LAMBDA", "OUTPUT_DIR"}),
    "e27_s6_entropy_acdc":      (f"{CFGDIR}/ctcmt_e27_s6_entropy_acdc.yaml",
                                 f"{CFGDIR}/ctcmt_e24_seghead_only_acdc.yaml",
                                 {"SOLVER.CTCMT_ENTROPY_WEIGHTED_CE", "OUTPUT_DIR"}),
    "e28_adaptive_routing_acdc": (f"{CFGDIR}/ctcmt_e28_adaptive_routing_acdc.yaml",
                                  f"{CFGDIR}/ctcmt_e26_partial025_acdc.yaml",
                                  {"SOLVER.CTCMT_ADAPTIVE_ROUTING", "OUTPUT_DIR"}),
}
print("OPTIONS O1-O3 (each vs its own parent)")
for name, (path, parent, allowed) in OPTION_ARMS.items():
    cfg, par = flatten(load(path)), flatten(load(parent))
    diff = {k for k in set(cfg) | set(par)
            if cfg.get(k, "<missing>") != par.get(k, "<missing>")}
    print(f"--- {name}  (parent {os.path.basename(parent)})")
    for k in sorted(diff):
        mark = "  " if k in allowed else "!!"
        print(f"  {mark} {k}: {par.get(k, '<missing>')!r} -> {cfg.get(k, '<missing>')!r}")
    if diff - allowed:
        failures.append(f"{name}: unintended diff vs parent: {sorted(diff - allowed)}")
    if cfg["SOLVER.CTCMT_CONFLICT_MODE"] != "aux_head_only":
        failures.append(f"{name}: must stay on the aux_head_only routing path")
    if cfg["MODEL.WEIGHTS"] != par["MODEL.WEIGHTS"]:
        failures.append(f"{name}: source checkpoint changed")
    lam = cfg["SOLVER.CTCMT_AUX_TRUNK_LAMBDA"]
    if not 0.0 <= lam <= 1.0:
        failures.append(f"{name}: lambda {lam} outside [0, 1]")
    print()

if failures:
    print("FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("All five arms differ from E13a in exactly one intended factor.")
print("Invariants held: seed=0, THRESHOLD_MAX=0.80, CTPV=False, det_only=False,")
print("CT-CR=soft_seg_global/floor 0.2, strong aug on, class-balanced CE on,")
print("fisher restore off, proto off, same source checkpoint, unique OUTPUT_DIRs.")
