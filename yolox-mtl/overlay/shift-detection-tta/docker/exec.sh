#!/usr/bin/env bash
# Exec into an already-running shift-detection-tta container.
# Usage: docker/exec.sh [command...]
set -euo pipefail
NAME="${NAME:-shift-detection-tta}"
if [[ $# -eq 0 ]]; then
    exec docker exec -it "$NAME" bash
else
    exec docker exec -it "$NAME" "$@"
fi
