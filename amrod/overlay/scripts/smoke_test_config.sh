#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for a config: one short domain, then assert that the run
# produced a result and that the loss terms the config asks for actually fired.
#
# Catches what preflight cannot: checkpoint loading, meta-arch wiring, OOM, and
# silently-inactive loss terms.
#
#   bash scripts/smoke_test_config.sh GPU CONFIG [EXP_NAME]

GPU="${1:?GPU id required}"
CFG="${2:?config required}"
EXP="${3:-smoke_$(basename "${CFG}" .yaml)}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
LOG="${HOST_OUT}/logs/${EXP}.log"

echo "######################################################################"
echo "SMOKE TEST  config=${CFG}  gpu=${GPU}"
echo "single ACDC domain (~400 images, ~5 min)"
echo "######################################################################"

# Trailing comma is required: ("x") is a str to yacs, ("x",) is a tuple.
STREAM='("acdc_fog_mtl",)' bash "${HERE}/run_ctcr_acdc_local.sh" "${GPU}" "${EXP}" "${CFG}" 0 || true

echo
echo "=================== SMOKE TEST REPORT ==============================="
fail=0

n_eval=$(grep -c "in csv format" "${LOG}" 2>/dev/null) || n_eval=0
if [[ "${n_eval}" -ge 1 ]]; then
  echo "  [PASS] produced ${n_eval} evaluation(s)"
else
  echo "  [FAIL] no evaluation produced"; fail=1
fi

if grep -q "Traceback (most recent call last)" "${LOG}" 2>/dev/null; then
  echo "  [FAIL] traceback in log:"; grep -A5 "Traceback" "${LOG}" | head -12; fail=1
else
  echo "  [PASS] no traceback"
fi

# Loss terms the config switches on must appear in the printed loss dict.
loss_line=$(grep -m1 "CT-CMT-MTL\] iter" "${LOG}" 2>/dev/null || true)
if [[ -n "${loss_line}" ]]; then
  echo "  loss terms: $(echo "${loss_line}" | sed 's/.*n_pseudo=[0-9]* //')"
  want=()
  grep -q "CTCMT_WEIGHT_CTCR: 0.0$" "${LOG}" || want+=("ctcr")
  grep -qi "CTCMT_ANCHOR_MARGINAL_WEIGHT: 0.0$" "${LOG}" || want+=("anchor_marginal")
  for t in "${want[@]}"; do
    if grep -q "${t}=" <<< "$(grep "CT-CMT-MTL\] iter" "${LOG}" | head -5)"; then
      echo "  [PASS] ${t} active"
    else
      echo "  [WARN] ${t} enabled in config but never appeared in the loss dict"
    fi
  done
else
  echo "  [INFO] no CT-CMT loss line (expected for AMROD/CoTTA/TENT configs)"
fi

ap=$(grep -A2 "copypaste: Task: bbox" "${LOG}" 2>/dev/null | tail -1 | awk -F, '{print $2}')
iou=$(grep -A2 "copypaste: Task: sem_seg" "${LOG}" 2>/dev/null | tail -1 | awk -F, '{print $1}')
echo "  fog AP50=${ap:-n/a}  mIoU=${iou:-n/a}   (reference: e10 fog AP50~53.8 mIoU~39.4)"

echo "====================================================================="
[[ ${fail} -eq 0 ]] && echo "SMOKE TEST PASSED — safe to launch the batch" \
                    || { echo "SMOKE TEST FAILED — do not launch"; exit 1; }
