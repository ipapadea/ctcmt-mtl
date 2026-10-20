#!/usr/bin/env bash
set -euo pipefail
cd /home/ilias/AMROD

tmux new-session -d -s a2_mixed42 'cd /home/ilias/AMROD && bash scripts/run_a2_mixed_lt.sh 0 42'
tmux new-session -d -s a2_mixed123 'cd /home/ilias/AMROD && bash scripts/run_a2_mixed_lt.sh 1 123'
tmux new-session -d -s a2_acdc_fog 'cd /home/ilias/AMROD && bash -c "bash scripts/run_a2_acdc.sh 2 0 && bash scripts/run_a2_acdc.sh 2 42 && bash scripts/run_a2_acdc.sh 2 123 && bash scripts/run_a2_fogx10.sh 2 0"'

echo "GPU0: A2 mixed seed42"
echo "GPU1: A2 mixed seed123"
echo "GPU2: A2 ACDC 0 -> 42 -> 123 -> fogx10 seed0"
echo "GPU3 left free for A seed42 L40S hardware correction."
