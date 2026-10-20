#!/usr/bin/env bash
set -euo pipefail

# Canary for the specialist / source-model study, to run BEFORE the 50-eval
# streams. Two Cityscapes-C domains (1000 images) per arm, so every per-step
# path executes ~1000 times AND a domain boundary is crossed -- the dataloader
# teardown/rebuild and evaluator switch that a single-domain smoke never
# reaches. The 500-image smoke on 2026-09-12 was too short to catch a 1-in-300
# event; this is sized so no path is exercised fewer times than that.
#
#   bash scripts/canary_specialists.sh [GPU_A] [GPU_B]
#
# Env (gpu1 defaults):
#   HOST_REPO HOST_OUT CSC_ROOT CITYSCAPES_ROOT

GPU_A="${1:-0}"
GPU_B="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"

# Preflight: the source checkpoints are resolved INSIDE the container relative
# to HOST_OUT. If HOST_OUT is unset on this machine, docker silently creates an
# empty directory and every arm dies on a missing checkpoint after ~30 s of
# startup. Fail here instead, in one second.
for ckpt in mask_rcnn_R50_cityscapes semantic_R50_cityscapes; do
  if [[ ! -f "${HOST_OUT}/${ckpt}/model_final.pth" ]]; then
    echo "ERROR: missing ${HOST_OUT}/${ckpt}/model_final.pth" >&2
    echo "       HOST_OUT=${HOST_OUT} -- export it for this machine, or copy the" >&2
    echo "       specialist checkpoints over (see scripts/batch_st_specialists.sh)." >&2
    exit 2
  fi
done
echo "preflight OK: both specialist checkpoints found under ${HOST_OUT}"

echo "######################################################################"
echo "SPECIALIST CANARY   2 domains x 500 images per arm"
echo "  gpu ${GPU_A}: ST-D  Mask R-CNN R50-FPN  + our detection CTTA"
echo "  gpu ${GPU_B}: ST-S  Semantic FPN R50    + our semantic  CTTA"
echo "######################################################################"

STREAM='("fog","motion_blur")' bash "${HERE}/run_mixed_lt_local.sh" \
  "${GPU_A}" canary_std_mrcnn "${CFG}/ctcmt_std_mrcnn_cscLT.yaml" 0 >/dev/null 2>&1 &
pid_a=$!
STREAM='("fog_semseg","motion_blur_semseg")' bash "${HERE}/run_mixed_lt_local.sh" \
  "${GPU_B}" canary_sts_semfpn "${CFG}/ctcmt_sts_semfpn_cscLT.yaml" 0 >/dev/null 2>&1 &
pid_b=$!
wait "${pid_a}" || echo "!!! ST-D canary exited non-zero"
wait "${pid_b}" || echo "!!! ST-S canary exited non-zero"

fail=0
report () {   # name, expect_det, expect_seg
  local log="${HOST_OUT}/logs/$1.log" want_det="$2" want_seg="$3"
  echo; echo "--- $1"
  [[ -f "${log}" ]] || { echo "  [FAIL] no log"; fail=1; return; }

  if grep -q "Traceback (most recent call last)" "${log}"; then
    echo "  [FAIL] traceback:"; grep -A12 "Traceback" "${log}" | head -16; fail=1
  else echo "  [PASS] no traceback"; fi

  # A checkpoint from the wrong architecture still loads -- it just leaves most
  # of the model at its random init and adapts garbage for six hours. A healthy
  # load emits none of these lines.
  local bad; bad=$(grep -cE "not found in the checkpoint|not used by the model|Skip loading parameter|incompatible shapes" "${log}") || bad=0
  if [[ "${bad}" -eq 0 ]]; then
    echo "  [PASS] checkpoint matches the architecture (0 key mismatches)"
  else
    echo "  [FAIL] ${bad} checkpoint key mismatches -- wrong checkpoint for this arch:"
    grep -E "not found in the checkpoint|Skip loading parameter" "${log}" | head -3
    fail=1
  fi
  echo "  source: $(grep -m1 -oE '/workspace/output/[^ ]*model_final.pth' "${log}")"

  local n; n=$(grep -c "in csv format" "${log}") || n=0
  [[ "${n}" -eq 2 ]] && echo "  [PASS] both domains evaluated (${n}/2)" \
                     || { echo "  [FAIL] ${n}/2 evaluations"; fail=1; }

  local line; line=$(grep -m1 "CT-CMT-MTL\] iter" "${log}" || true)
  [[ -n "${line}" ]] && echo "  loss terms: ${line#*n_pseudo=}" \
                     || { echo "  [FAIL] adaptation never logged"; fail=1; }

  if [[ "${want_det}" == yes ]]; then
    grep -q "det/loss_cls" "${log}" && echo "  [PASS] detection loss active" \
      || { echo "  [FAIL] detection loss never fired"; fail=1; }
    grep -q "seg/soft_ce" "${log}" && { echo "  [FAIL] seg loss leaked into ST-D"; fail=1; } \
      || echo "  [PASS] no segmentation loss (det-only)"
  fi
  if [[ "${want_seg}" == yes ]]; then
    grep -q "seg/soft_ce" "${log}" && echo "  [PASS] segmentation loss active" \
      || { echo "  [FAIL] segmentation loss never fired"; fail=1; }
    grep -q "det/loss_cls" "${log}" && { echo "  [FAIL] detection loss leaked into ST-S"; fail=1; } \
      || echo "  [PASS] no detection loss (seg-only)"
    # The aug-average path is conditional; prove it actually ran.
    local a; a=$(grep -o "segaug=[0-9]*/[0-9]*" "${log}" | tail -1 || true)
    [[ -n "${a}" && "${a#segaug=}" != 0/* ]] \
      && echo "  [PASS] aug-averaging exercised (${a})" \
      || { echo "  [WARN] aug-averaging never triggered (${a:-absent})"; }
  fi

  local ap iou
  ap=$(grep -A2 "copypaste: Task: bbox" "${log}" | tail -1 | awk -F, '{print $2}')
  iou=$(grep -A2 "copypaste: Task: sem_seg" "${log}" | tail -1 | awk -F, '{print $1}')
  echo "  AP50=${ap:-n/a}  mIoU=${iou:-n/a}"
}

report canary_std_mrcnn  yes no
report canary_sts_semfpn no  yes

echo
echo "====================================================================="
[[ ${fail} -eq 0 ]] && echo "CANARY PASSED - safe to launch the x10 streams" \
                    || { echo "CANARY FAILED - do not launch"; exit 1; }
