#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the S1..S5 negative-transfer screening arms: one Cityscapes-C
# domain each (fog, 500 images, ~6 min), then assert that the mechanism each
# config claims to enable actually fired at runtime.
#
# Catches what the config diff cannot: meta-arch wiring, the extra backward
# passes, OOM from holding two gradient snapshots, and mechanisms that are
# switched on but never reached.
#
#   bash scripts/smoke_screening_arms.sh [GPU]

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"

ARMS=(
  "e16_protectedgrad:${CFG}/ctcmt_e16_protectedgrad.yaml:protect_det"
  "e17_cagrad:${CFG}/ctcmt_e17_cagrad.yaml:cagrad"
  "e18_harddecouple:${CFG}/ctcmt_e18_harddecouple.yaml:hard_decouple"
  "e19_frozentrunk:${CFG}/ctcmt_e19_frozentrunk.yaml:frozen"
  "e20_dynweight:${CFG}/ctcmt_e20_dynweight.yaml:dyn_weight"
)

fail=0
for arm in "${ARMS[@]}"; do
  IFS=: read -r name cfg mode <<< "${arm}"
  exp="smoke_${name}"
  log="${HOST_OUT}/logs/${exp}.log"
  echo "######################################################################"
  echo "SMOKE  ${name}   mode=${mode}   gpu=${GPU}"
  echo "######################################################################"
  STREAM='("fog_mtl",)' bash "${HERE}/run_mixed_lt_local.sh" \
    "${GPU}" "${exp}" "${cfg}" 0 || true

  echo
  echo "--- report: ${name}"
  if grep -q "Traceback (most recent call last)" "${log}" 2>/dev/null; then
    echo "  [FAIL] traceback"; grep -A8 "Traceback" "${log}" | head -14; fail=1
  else
    echo "  [PASS] no traceback"
  fi

  n_eval=$(grep -c "in csv format" "${log}" 2>/dev/null) || n_eval=0
  [[ "${n_eval}" -ge 1 ]] && echo "  [PASS] ${n_eval} evaluation(s)" \
                          || { echo "  [FAIL] no evaluation"; fail=1; }

  if [[ "${mode}" == "frozen" ]]; then
    if grep -q "FREEZE_SHARED_TRUNK" "${log}" 2>/dev/null; then
      echo "  [PASS] $(grep -m1 -o 'FREEZE_SHARED_TRUNK.*' "${log}")"
    else
      echo "  [FAIL] freeze never applied"; fail=1
    fi
    if grep -q "CT-CMT-GRAD\]" "${log}" 2>/dev/null; then
      echo "  [FAIL] gradient surgery ran in the frozen-trunk arm"; fail=1
    else
      echo "  [PASS] no gradient surgery (mode none, as intended)"
    fi
  else
    line=$(grep -m1 "CT-CMT-GRAD\]" "${log}" 2>/dev/null || true)
    if [[ -n "${line}" ]] && grep -q "mode=${mode}" <<< "${line}"; then
      echo "  [PASS] ${line#*\] }"
      grep "CT-CMT-GRAD\]" "${log}" | tail -1 | sed 's/^/         last: /'
    else
      echo "  [FAIL] no [CT-CMT-GRAD] line with mode=${mode}"; fail=1
    fi
    dg=$(grep -m1 "CT-CMT-GRADDIAG\]" "${log}" 2>/dev/null || true)
    [[ -n "${dg}" ]] && echo "  [PASS] ${dg#*\] }" \
                     || { echo "  [FAIL] per-component diagnostics never logged"; fail=1; }
  fi

  ap=$(grep -A2 "copypaste: Task: bbox" "${log}" 2>/dev/null | tail -1 | awk -F, '{print $2}')
  iou=$(grep -A2 "copypaste: Task: sem_seg" "${log}" 2>/dev/null | tail -1 | awk -F, '{print $1}')
  echo "  fog AP50=${ap:-n/a}  mIoU=${iou:-n/a}"
  echo
done

echo "====================================================================="
[[ ${fail} -eq 0 ]] && echo "ALL SMOKE TESTS PASSED - safe to launch the batch" \
                    || { echo "SMOKE TESTS FAILED - do not launch"; exit 1; }
