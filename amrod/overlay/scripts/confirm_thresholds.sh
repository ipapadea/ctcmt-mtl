#!/usr/bin/env bash
set -euo pipefail

# Confirm (or refute) the threshold-saturation explanation for the Cityscapes-C
# detection plateau, in ~13 minutes.
#
# Diagnosis so far: our pseudo-label count saturates at ~5.3/image from round 3
# and AP50 stops improving at round 5, while AMROD's dynamic thresholds keep
# falling (0.867 -> 0.816) and its step rate keeps rising (0.72 -> 0.96).
#
# The threshold rule is  tau <- gamma*tau + (1-gamma)*alpha*sqrt(mean_score),
# clipped to [THRESHOLD_MINI, THRESHOLD_MAX] = [0.7, 0.9], alpha = 1.3.
# With our mean teacher score ~0.69 the fixed point is 1.3*sqrt(0.69) ~ 1.08,
# i.e. above the ceiling -- so tau should sit pinned at exactly 0.900.
#
# PASS (hypothesis confirmed): mean_thr converges to ~0.900 and stays there.
# FAIL (hypothesis refuted):   mean_thr settles below 0.9 and keeps moving.
#
#   bash scripts/confirm_thresholds.sh [GPU] [CONFIG]

GPU="${1:-1}"
CFG="${2:-detectron2/configs/Cityscapes/ctcmt_e11_bothsc_ctcrD.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
EXP="thrcheck_$(basename "${CFG}" .yaml | sed 's/^ctcmt_//')"
LOG="${HOST_OUT}/logs/${EXP}.log"

echo "######################################################################"
echo "THRESHOLD SATURATION CHECK   gpu=${GPU}"
echo "2 corruptions (fog, motion_blur) = 1000 images, ~13 min"
echo "######################################################################"

STREAM='("fog_mtl","motion_blur_mtl")' \
  bash "${HERE}/run_mixed_lt_local.sh" "${GPU}" "${EXP}" "${CFG}" 0 || true

echo
echo "=================== THRESHOLD TRACE ================================="
grep -o "iter=[0-9]* score_em=[0-9.]* n_pseudo=[0-9]* thr=[0-9./]*" "${LOG}" \
  | awk 'NR%2==1' | head -20 || echo "  no thr= field found -- is the logging patch applied?"

echo
last=$(grep -o "thr=[0-9.]*/[0-9.]*/[0-9.]*" "${LOG}" | tail -1 || true)
echo "final ${last:-<none>}   (format: min/mean/max over the 8 detection classes)"

mn=$(echo "${last}" | sed 's/thr=//' | cut -d/ -f1)
mu=$(echo "${last}" | sed 's/thr=//' | cut -d/ -f2)
echo
echo "AMROD reference on the same protocol: mean_thr 0.867 (R1) -> 0.816 (R10)"
if [[ -n "${mu}" ]] && awk "BEGIN{exit !(${mu:-0} >= 0.895)}"; then
  echo ">>> FULLY SATURATED: the MEAN threshold sits at the ceiling."
  echo "    Sweep THRESHOLD_MAX (0.9 -> 0.85 / 0.8) or ALPHA_DT (1.3 -> 1.1)."
elif [[ -n "${mn}" ]] && awk "BEGIN{exit !(${mu:-0} >= 0.84)}"; then
  echo ">>> PARTIAL: some classes pinned at 0.9, mean ${mu} still below the ceiling."
  echo "    This matches AMROD at round 1 (0.867), so the starting point is NOT the"
  echo "    differentiator. The hypothesis is about the TRAJECTORY over 10 rounds,"
  echo "    which a 2-domain run cannot show. Run the intervention instead."
else
  echo ">>> NOT SATURATED (mean ${mu}). Threshold saturation is not the cause."
fi
echo "====================================================================="
echo "full log: ${LOG}"
