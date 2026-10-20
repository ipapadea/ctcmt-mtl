#!/usr/bin/env bash
# Enhancement matrix — E1..E6 x 3 seeds = 18 runs on 4 GPUs sequentially.
# GPU 0: E1 seed0 -> E5 seed0 -> (5 runs)
# GPU 1: E1 seed42 -> E5 seed42 -> (5 runs)
# GPU 2: E1 seed123 -> E5 seed123 -> (5 runs)
# GPU 3: E6 all 3 seeds
# Total wall time: ~5 * 15 min = ~75 min per GPU.
set -euo pipefail

SESSION="enh_matrix"
AMROD=/home/ilias/AMROD

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} exists; run:  tmux kill-session -t ${SESSION}"
  exit 1
fi

LOGDIR=/data/ilias/panoptic_fpn/output/ctta_acdc/logs_enhancements
mkdir -p "${LOGDIR}"

# Each GPU runs its 5 (or 3) tracks sequentially in the same window.
# Names inside the run function keep the loop compact.

tmux new-session -d -s "${SESSION}" -n gpu0 "bash -c '
  cd ${AMROD}
  for E in e1_ctpv e2_entropy_ce e3_teacher_aug e4_dir_gate e5_adaptive_str; do
    echo \"[gpu0] \${E} seed0\"
    bash scripts/run_ctta_acdc.sh 0 ctcmt_\${E}_seed0 \
      2>&1 | tee ${LOGDIR}/\${E}_seed0.log
  done
  echo \"[gpu0] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu1 "bash -c '
  cd ${AMROD}
  for E in e1_ctpv e2_entropy_ce e3_teacher_aug e4_dir_gate e5_adaptive_str; do
    echo \"[gpu1] \${E} seed42\"
    bash scripts/run_ctta_acdc.sh 1 ctcmt_\${E}_seed42 \
      2>&1 | tee ${LOGDIR}/\${E}_seed42.log
  done
  echo \"[gpu1] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu2 "bash -c '
  cd ${AMROD}
  for E in e1_ctpv e2_entropy_ce e3_teacher_aug e4_dir_gate e5_adaptive_str; do
    echo \"[gpu2] \${E} seed123\"
    bash scripts/run_ctta_acdc.sh 2 ctcmt_\${E}_seed123 \
      2>&1 | tee ${LOGDIR}/\${E}_seed123.log
  done
  echo \"[gpu2] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu3 "bash -c '
  cd ${AMROD}
  for SEED in 0 42 123; do
    echo \"[gpu3] e6_all seed\${SEED}\"
    bash scripts/run_ctta_acdc.sh 3 ctcmt_e6_all_seed\${SEED} \
      2>&1 | tee ${LOGDIR}/e6_all_seed\${SEED}.log
  done
  echo \"[gpu3] DONE\"; exec bash'"

echo "started tmux session: ${SESSION}"
echo "attach:   tmux attach -t ${SESSION}"
echo "windows:  tmux list-windows -t ${SESSION}"
echo "logs:     ${LOGDIR}/"
