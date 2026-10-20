#!/usr/bin/env bash
# Loopback matrix: after fog→night→rain→snow, adapt one more step on the
# original Cityscapes val set. Reports mAP_Loopback which we compare with
# mAP_Source to obtain the Forgetting metric (Moraiti et al. EJAI 2026).
#
# GPU 0: MTL v2+ctpv seed0    -> Det MR seed0
# GPU 1: MTL v2+ctpv seed42   -> Det MR seed42
# GPU 2: MTL v2+ctpv seed123  -> Det MR seed123
# GPU 3: idle
set -euo pipefail

SESSION="loopback"
AMROD=/home/ilias/AMROD

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} exists; run:  tmux kill-session -t ${SESSION}"
  exit 1
fi

LOGDIR=/data/ilias/panoptic_fpn/output/ctta_acdc/logs_loopback
mkdir -p "${LOGDIR}"

tmux new-session -d -s "${SESSION}" -n gpu0 "bash -c '
  cd ${AMROD}
  echo \"[gpu0] MTL v2+ctpv seed0 (+loopback)\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_loopback_v2_ctpv_seed0 \
    2>&1 | tee ${LOGDIR}/mtl_v2_ctpv_seed0.log
  echo \"[gpu0] Det MR seed0 (+loopback)\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_loopback_det_mr_seed0 \
    2>&1 | tee ${LOGDIR}/det_mr_seed0.log
  echo \"[gpu0] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu1 "bash -c '
  cd ${AMROD}
  echo \"[gpu1] MTL v2+ctpv seed42 (+loopback)\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_loopback_v2_ctpv_seed42 \
    2>&1 | tee ${LOGDIR}/mtl_v2_ctpv_seed42.log
  echo \"[gpu1] Det MR seed42 (+loopback)\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_loopback_det_mr_seed42 \
    2>&1 | tee ${LOGDIR}/det_mr_seed42.log
  echo \"[gpu1] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu2 "bash -c '
  cd ${AMROD}
  echo \"[gpu2] MTL v2+ctpv seed123 (+loopback)\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_loopback_v2_ctpv_seed123 \
    2>&1 | tee ${LOGDIR}/mtl_v2_ctpv_seed123.log
  echo \"[gpu2] Det MR seed123 (+loopback)\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_loopback_det_mr_seed123 \
    2>&1 | tee ${LOGDIR}/det_mr_seed123.log
  echo \"[gpu2] DONE\"; exec bash'"

echo "started tmux session: ${SESSION}"
echo "attach:   tmux attach -t ${SESSION}"
echo "logs:     ${LOGDIR}/"
