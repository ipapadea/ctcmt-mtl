#!/usr/bin/env bash
# Loopback Option B: AMROD + CoTTA semseg, one run each.
# Measures Forgetting = mX_Source - mX_Loopback for comparison methods.
#
# GPU 0: AMROD loopback  (det, CityscapesInstanceEvaluator on 5th domain)
# GPU 1: CoTTA semseg loopback (seg, CityscapesSemanticEvaluator on 5th domain)
set -euo pipefail

SESSION="loopback_b"
AMROD=/home/ilias/AMROD

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} exists; run:  tmux kill-session -t ${SESSION}"
  exit 1
fi

LOGDIR=/data/ilias/panoptic_fpn/output/ctta_acdc/logs_loopback
mkdir -p "${LOGDIR}"

tmux new-session -d -s "${SESSION}" -n amrod "bash -c '
  cd ${AMROD}
  echo \"[gpu0] AMROD loopback\"
  bash scripts/run_ctta_acdc.sh 0 amrod_loopback \
    2>&1 | tee ${LOGDIR}/amrod_loopback.log
  echo \"[gpu0] DONE\"; exec bash'"

tmux new-window -t "${SESSION}" -n cotta_semseg "bash -c '
  cd ${AMROD}
  echo \"[gpu1] CoTTA semseg loopback\"
  bash scripts/run_ctta_acdc.sh 1 cotta_semseg_loopback \
    2>&1 | tee ${LOGDIR}/cotta_semseg_loopback.log
  echo \"[gpu1] DONE\"; exec bash'"

echo "started tmux session: ${SESSION}"
echo "attach:   tmux attach -t ${SESSION}"
echo "logs:     ${LOGDIR}/amrod_loopback.log"
echo "          ${LOGDIR}/cotta_semseg_loopback.log"
