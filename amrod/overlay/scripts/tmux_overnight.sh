#!/usr/bin/env bash
# Overnight orchestrator: runs 6 multi-seed CT-CL vs V2 configs across 3 GPUs
# and (optionally) the W3TTA-OD baseline on GPU 3.
#
# Each pane launches sequentially: seed A -> seed B on that GPU.
# GPU 0: CT-CL noV2 seed0 -> V2 seed0
# GPU 1: CT-CL noV2 seed42 -> V2 seed42
# GPU 2: CT-CL noV2 seed123 -> V2 seed123
# GPU 3: W3TTA-OD source stats -> W3TTA-OD ACDC
set -euo pipefail

SESSION="ctta_multiseed"
AMROD=/home/ilias/AMROD
W3TTA=/home/ilias/vgcmt-baselines/w3ttaod

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session ${SESSION} exists; run:  tmux kill-session -t ${SESSION}"
  exit 1
fi

mkdir -p /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed

tmux new-session -d -s "${SESSION}" -n gpu0 "bash -c '
  cd ${AMROD}
  echo \"[gpu0] seed0 CT-CL no-V2\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_mtl_ctcl_no_v2_seed0 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/ctcl_noV2_seed0.log
  echo \"[gpu0] seed0 V2\"
  bash scripts/run_ctta_acdc.sh 0 ctcmt_v2_seed0 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/V2_seed0.log
  echo \"[gpu0] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu1 "bash -c '
  cd ${AMROD}
  echo \"[gpu1] seed42 CT-CL no-V2\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_mtl_ctcl_no_v2_seed42 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/ctcl_noV2_seed42.log
  echo \"[gpu1] seed42 V2\"
  bash scripts/run_ctta_acdc.sh 1 ctcmt_v2_seed42 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/V2_seed42.log
  echo \"[gpu1] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu2 "bash -c '
  cd ${AMROD}
  echo \"[gpu2] seed123 CT-CL no-V2\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_mtl_ctcl_no_v2_seed123 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/ctcl_noV2_seed123.log
  echo \"[gpu2] seed123 V2\"
  bash scripts/run_ctta_acdc.sh 2 ctcmt_v2_seed123 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/V2_seed123.log
  echo \"[gpu2] DONE\"; exec bash'"

tmux new-window  -t "${SESSION}" -n gpu3 "bash -c '
  cd ${W3TTA}
  echo \"[gpu3] W3TTA-OD source stats\"
  bash scripts/collect_source_stats_acdc.sh 3 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/w3ttaod_source_stats.log \
    || { echo \"[gpu3] SOURCE STATS FAILED\"; exec bash; }
  echo \"[gpu3] W3TTA-OD ACDC continual\"
  bash scripts/run_acdc.sh 3 \
    2>&1 | tee /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/w3ttaod_acdc.log
  echo \"[gpu3] DONE\"; exec bash'"

echo "started tmux session: ${SESSION}"
echo "attach:   tmux attach -t ${SESSION}"
echo "windows:  tmux list-windows -t ${SESSION}"
echo "logs:     /data/ilias/panoptic_fpn/output/ctta_acdc/logs_multiseed/"
