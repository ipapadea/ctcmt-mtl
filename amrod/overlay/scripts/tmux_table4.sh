#!/usr/bin/env bash
# Table 4 orchestrator: fills the missing rows with 3 seeds each.
#   MTL no-CL no-V2   x {seed0, seed42, seed123}
#   MTL no-CL + V2    x {seed0, seed42, seed123}
#
# GPU 0: no-CL no-V2 seed0   -> no-CL + V2 seed0
# GPU 1: no-CL no-V2 seed42  -> no-CL + V2 seed42
# GPU 2: no-CL no-V2 seed123 -> no-CL + V2 seed123
# GPU 3: idle (reserved for other work)
set -euo pipefail

SESSION="table4_multiseed"
AMROD=/home/ilias/AMROD

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} exists; run:  tmux kill-session -t ${SESSION}"
  exit 1
fi

LOGDIR=/data/ilias/panoptic_fpn/output/ctta_acdc/logs_table4
mkdir -p "${LOGDIR}"

tmux new-session -d -s "${SESSION}" -n gpu0 "bash -c '
  cd ${AMROD}
  echo \"[gpu0] seed0 no-CL no-V2\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_mtl_no_ctcl_seed0 \
    2>&1 | tee ${LOGDIR}/no_ctcl_seed0.log
  echo \"[gpu0] seed0 no-CL + V2\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_mtl_no_ctcl_v2_seed0 \
    2>&1 | tee ${LOGDIR}/no_ctcl_v2_seed0.log
  echo \"[gpu0] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu1 "bash -c '
  cd ${AMROD}
  echo \"[gpu1] seed42 no-CL no-V2\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_mtl_no_ctcl_seed42 \
    2>&1 | tee ${LOGDIR}/no_ctcl_seed42.log
  echo \"[gpu1] seed42 no-CL + V2\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_mtl_no_ctcl_v2_seed42 \
    2>&1 | tee ${LOGDIR}/no_ctcl_v2_seed42.log
  echo \"[gpu1] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu2 "bash -c '
  cd ${AMROD}
  echo \"[gpu2] seed123 no-CL no-V2\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_mtl_no_ctcl_seed123 \
    2>&1 | tee ${LOGDIR}/no_ctcl_seed123.log
  echo \"[gpu2] seed123 no-CL + V2\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_mtl_no_ctcl_v2_seed123 \
    2>&1 | tee ${LOGDIR}/no_ctcl_v2_seed123.log
  echo \"[gpu2] DONE\"; exec bash'"

echo "started tmux session: ${SESSION}"
echo "attach:   tmux attach -t ${SESSION}"
echo "windows:  tmux list-windows -t ${SESSION}"
echo "logs:     ${LOGDIR}/"
