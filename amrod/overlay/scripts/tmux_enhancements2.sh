#!/usr/bin/env bash
# Follow-up matrix: E7 (E1+E3 winners) + E4b (directional gate w/ boost=1.5)
# 6 runs distributed for maximum parallelism:
#   GPU 0: E7 seed0    -> E4b seed0
#   GPU 1: E7 seed42   -> E4b seed42
#   GPU 2: E7 seed123  -> E4b seed123
#   GPU 3: idle (or can be filled with something else if you like)
# Wall time ~30 min (2 x ~15 min per GPU).
set -euo pipefail

SESSION="enh_matrix2"
AMROD=/home/ilias/AMROD

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} exists; run:  tmux kill-session -t ${SESSION}"
  exit 1
fi

LOGDIR=/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements2
mkdir -p "${LOGDIR}"

tmux new-session -d -s "${SESSION}" -n gpu0 "bash -c '
  cd ${AMROD}
  echo \"[gpu0] E7 seed0\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_e7_e1_e3_seed0 \
    2>&1 | tee ${LOGDIR}/e7_seed0.log
  echo \"[gpu0] E4b seed0\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_e4b_dir_gate_boost15_seed0 \
    2>&1 | tee ${LOGDIR}/e4b_seed0.log
  echo \"[gpu0] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu1 "bash -c '
  cd ${AMROD}
  echo \"[gpu1] E7 seed42\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_e7_e1_e3_seed42 \
    2>&1 | tee ${LOGDIR}/e7_seed42.log
  echo \"[gpu1] E4b seed42\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_e4b_dir_gate_boost15_seed42 \
    2>&1 | tee ${LOGDIR}/e4b_seed42.log
  echo \"[gpu1] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu2 "bash -c '
  cd ${AMROD}
  echo \"[gpu2] E7 seed123\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_e7_e1_e3_seed123 \
    2>&1 | tee ${LOGDIR}/e7_seed123.log
  echo \"[gpu2] E4b seed123\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_e4b_dir_gate_boost15_seed123 \
    2>&1 | tee ${LOGDIR}/e4b_seed123.log
  echo \"[gpu2] DONE\"; exec bash'"

echo "started tmux session: ${SESSION}"
echo "attach:   tmux attach -t ${SESSION}"
echo "windows:  tmux list-windows -t ${SESSION}"
echo "logs:     ${LOGDIR}/"
