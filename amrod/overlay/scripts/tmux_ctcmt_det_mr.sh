#!/usr/bin/env bash
# CTCMT_Det on Mask R-CNN source — 3 seeds sequentially on GPU 3.
# Leaves GPUs 0/1/2 free (e.g. for the ongoing table4_multiseed session).
# Total wall time ~45 min (3 seeds x ~15 min).
set -euo pipefail

SESSION="ctcmt_det_mr"
GPU=${1:-3}
AMROD=/home/ilias/AMROD

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} exists; run:  tmux kill-session -t ${SESSION}"
  exit 1
fi

LOGDIR=/data/ilias/panoptic_fpn/output/ctta_acdc/logs_ctcmt_det_mr
mkdir -p "${LOGDIR}"

tmux new-session -d -s "${SESSION}" -n gpu${GPU} "bash -c '
  cd ${AMROD}
  for SEED in 0 42 123; do
    echo \"[gpu${GPU}] CTCMT_Det MR seed\${SEED}\"
    bash scripts/run_ctta_acdc.sh ${GPU} ctcmt_det_mr_seed\${SEED} 2>&1 | tee ${LOGDIR}/seed\${SEED}.log
  done
  echo \"[gpu${GPU}] ALL SEEDS DONE\"; exec bash'"

echo "started tmux session: ${SESSION} (on GPU ${GPU})"
echo "attach:   tmux attach -t ${SESSION}"
echo "logs:     ${LOGDIR}/"

