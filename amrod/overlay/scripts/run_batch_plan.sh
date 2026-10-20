#!/usr/bin/env bash
set -euo pipefail

# Prepared batch. NOTHING RUNS unless a stage is named explicitly.
#
#   bash scripts/run_batch_plan.sh list
#   bash scripts/run_batch_plan.sh acdc   GPU        # cheap screen, ~25 min/run
#   bash scripts/run_batch_plan.sh cscA   GPU        # CS-C LT slot A, ~8 h/run
#   bash scripts/run_batch_plan.sh cscB   GPU        # CS-C LT slot B, ~8 h/run
#
# Rationale for the split: the class-collapse countermeasures only pay off after
# round 3, so ACDC (single 4-domain pass) cannot validate them -- it only shows
# their cost. They must be judged on the Cityscapes-C x10 stream.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
STAGE="${1:-list}"
GPU="${2:-0}"

acdc()  { bash "${HERE}/run_ctcr_acdc_local.sh" "${GPU}" "$1" "${CFG}/$2" "${3:-0}"; }
csclt() { bash "${HERE}/run_mixed_lt_local.sh"  "${GPU}" "$1" "${CFG}/$2" "${3:-0}"; }

case "${STAGE}" in
list)
  cat <<'TXT'
STAGE acdc   (~25 min each)  -- cheap screen, expect NEUTRAL/slightly negative
                                for the E10 arms; they exist to catch regressions
  e9_fisher_acdc_s0        ctcmt_e9_fisher.yaml            [DIAGNOSTIC, AMROD's]
  e10_clsbal_acdc_s0       ctcmt_e10_clsbal.yaml
  e10_clsbalsc_acdc_s0     ctcmt_e10_clsbal_scaled.yaml
  e10_anchormarg_acdc_s0   ctcmt_e10_anchormarg.yaml
  e10_bothsc_acdc_s0       ctcmt_e10_both_scaled.yaml

STAGE cscA   (~8 h each)     -- the decisive comparison
  e9_strongaug_cscLT_s0    ctcmt_e9_strongaug.yaml         [NEW BASELINE, required]
  e10_bothsc_cscLT_s0      ctcmt_e10_both_scaled.yaml      [the fix]

STAGE cscB   (~8 h each)     -- decomposition, only if cscA shows the fix works
  e10_clsbalsc_cscLT_s0    ctcmt_e10_clsbal_scaled.yaml
  e10_anchormarg_cscLT_s0  ctcmt_e10_anchormarg.yaml

Already done, no need to rerun:
  A / A2 / B / C on CS-C LT       (old base, no strong aug)
  D on ACDC 3 seeds               37.074 +/- 0.120 AP50 | 35.017 +/- 0.069 mIoU
  e9_strongaug, e9_fisher on ACDC 38.194 / 37.780 AP50
  source-only PFN on CS-C         12.70 AP50 / 26.25 mIoU (12 corr)
  AMROD / CoTTA / TENT on PFN, ACDC seed 0
TXT
  ;;
acdc)
  acdc e9_fisher_acdc_s0       ctcmt_e9_fisher.yaml
  acdc e10_clsbal_acdc_s0      ctcmt_e10_clsbal.yaml
  acdc e10_clsbalsc_acdc_s0    ctcmt_e10_clsbal_scaled.yaml
  acdc e10_anchormarg_acdc_s0  ctcmt_e10_anchormarg.yaml
  acdc e10_bothsc_acdc_s0      ctcmt_e10_both_scaled.yaml
  ;;
cscA)
  csclt e9_strongaug_cscLT_s0  ctcmt_e9_strongaug.yaml
  csclt e10_bothsc_cscLT_s0    ctcmt_e10_both_scaled.yaml
  ;;
cscB)
  csclt e10_clsbalsc_cscLT_s0  ctcmt_e10_clsbal_scaled.yaml
  csclt e10_anchormarg_cscLT_s0 ctcmt_e10_anchormarg.yaml
  ;;
*)
  echo "unknown stage '${STAGE}' -- run: bash $0 list" >&2; exit 2 ;;
esac
