#!/usr/bin/env python3
"""Pre-flight check for the specialist / source-model study (ST-D, ST-S).

These two arms deliberately change the source checkpoint, so they are NOT
same-source with E13a/E15/E21 and must be reported separately. What this script
guarantees is that *nothing else* changed: the adaptation recipe must be
identical to the same-task PanopticFPN arm it is compared against.

  ST-D  vs  E15 (det-only on the MTL checkpoint)
  ST-S  vs  E21 (seg-only on the MTL checkpoint)

  python scripts/verify_specialist_configs.py
"""
import sys

from detectron2.config import get_cfg

CFGDIR = "detectron2/configs/Cityscapes"

PAIRS = [
    ("ST-D", f"{CFGDIR}/ctcmt_std_mrcnn_cscLT.yaml",
     f"{CFGDIR}/ctcmt_e15_detonly_thr080.yaml", "GeneralizedRCNN",
     "/workspace/output/mask_rcnn_R50_cityscapes/model_final.pth"),
    ("ST-S", f"{CFGDIR}/ctcmt_sts_semfpn_cscLT.yaml",
     f"{CFGDIR}/ctcmt_e21_segonly_thr080.yaml", "SemanticSegmentor",
     "/workspace/output/semantic_R50_cityscapes/model_final.pth"),
]

# SOLVER keys that are allowed to differ: schedule knobs that are meaningless in
# --eval-only mode, and the single-task switches themselves.
SOLVER_EXEMPT = {
    "STEPS", "MAX_ITER", "WARMUP_ITERS", "CHECKPOINT_PERIOD", "IMS_PER_BATCH",
    "CTCMT_DET_ONLY", "CTCMT_SEG_ONLY",
}

CYCLE = ("fog", "motion_blur", "snow", "brightness", "defocus_blur")


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
    cfg.merge_from_list(["SEED", "0"])
    return cfg


failures = []
for name, spec_path, ref_path, want_arch, want_weights in PAIRS:
    spec, ref = load(spec_path), load(ref_path)
    fs, fr = flatten(spec.SOLVER), flatten(ref.SOLVER)

    print(f"--- {name}  {spec_path}")
    print(f"      reference recipe : {ref_path}")
    print(f"      student meta-arch: {spec.MODEL.CTCMT_STUDENT_META_ARCH}")
    print(f"      source checkpoint: {spec.MODEL.WEIGHTS}")

    diff = sorted(k for k in set(fs) | set(fr)
                  if fs.get(k, "<missing>") != fr.get(k, "<missing>"))
    for k in diff:
        tag = "  " if k in SOLVER_EXEMPT else "!!"
        print(f"      {tag} SOLVER.{k}: {fr.get(k,'<missing>')!r} -> {fs.get(k,'<missing>')!r}")
    unexpected = [k for k in diff if k not in SOLVER_EXEMPT]
    if unexpected:
        failures.append(f"{name}: adaptation recipe differs from {ref_path}: {unexpected}")

    if spec.MODEL.CTCMT_STUDENT_META_ARCH != want_arch:
        failures.append(f"{name}: student meta-arch is {spec.MODEL.CTCMT_STUDENT_META_ARCH}")
    if spec.MODEL.WEIGHTS != want_weights:
        failures.append(f"{name}: wrong source checkpoint {spec.MODEL.WEIGHTS}")
    if spec.MODEL.WEIGHTS == ref.MODEL.WEIGHTS:
        failures.append(f"{name}: checkpoint did NOT change -- this is not a specialist run")
    if spec.MODEL.META_ARCHITECTURE != "CTCMT_MTL":
        failures.append(f"{name}: meta-arch must stay CTCMT_MTL")

    # The stream must cover the five long-term domains, in cycle order, with the
    # evaluator suffix that matches what the student can actually emit.
    suffix = "" if name == "ST-D" else "_semseg"
    want = tuple(f"{c}{suffix}" for c in CYCLE)
    if tuple(spec.DATASETS.TEST) != want:
        failures.append(f"{name}: DATASETS.TEST is {tuple(spec.DATASETS.TEST)}, expected {want}")

    for k in ("CTCMT_CTPV_ENABLED", "CTCMT_PROTO_ANCHOR", "CTCMT_FISHER_RESTORE"):
        if getattr(spec.SOLVER, k):
            failures.append(f"{name}: {k} must be False")
    if spec.SOLVER.CTCMT_CONFLICT_MODE != "none":
        failures.append(f"{name}: CTCMT_CONFLICT_MODE must be 'none'")
    if not spec.SOLVER.CTCMT_STRONG_AUG_STUDENT:
        failures.append(f"{name}: strong augmentation must stay on")
    if spec.SOLVER.THRESHOLD_MAX != 0.80:
        failures.append(f"{name}: THRESHOLD_MAX must be 0.80")
    print()

if failures:
    print("FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("Both specialist arms reproduce their same-source reference recipe exactly;")
print("the only intended differences are the student meta-arch, the source")
print("checkpoint and the evaluator-appropriate dataset names.")
print()
print("REMINDER: these change the source model. Report them as a separate")
print("specialist/source-model study, never inside the same-source table.")
