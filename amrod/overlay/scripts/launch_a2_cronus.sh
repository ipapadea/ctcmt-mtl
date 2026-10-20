#!/usr/bin/env bash
set -euo pipefail
cd /home/ilias/AMROD
tmux new-session -d -s a2_mixed0 'cd /home/ilias/AMROD && bash scripts/run_a2_mixed_lt.sh 0 0'
echo "Cronus GPU0: A2 mixed seed0"
