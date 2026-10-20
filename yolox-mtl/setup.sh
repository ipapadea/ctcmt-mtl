#!/usr/bin/env bash
# Build the YOLOX-MTL working tree: panagiotamoraiti/shift-tta at a pinned commit + our patch + our files.
#
#   bash setup.sh TARGET_DIR
#
# Our changes to eight upstream files are in patches/shift_tta_upstream_changes.patch, all other files of ours are
# new files under overlay/.
set -euo pipefail
TARGET="${1:?usage: bash setup.sh TARGET_DIR}"
UPSTREAM="${UPSTREAM:-https://github.com/panagiotamoraiti/shift-tta.git}"
BASE="${BASE:-a2449e42fc3d09a038228e9f6232863ee01e8ece}"   # upstream main our work is based on
HERE="$(cd "$(dirname "$0")" && pwd)"

[[ -e "$TARGET" ]] && { echo "ERROR: $TARGET already exists" >&2; exit 2; }
mkdir -p "$TARGET" && cd "$TARGET"
git init -q . && git remote add origin "$UPSTREAM"
git fetch -q --depth 1 origin "$BASE" && git checkout -q FETCH_HEAD
git apply --whitespace=nowarn "$HERE/patches/shift_tta_upstream_changes.patch"
cp -a "$HERE/overlay/." .
echo "OK: $TARGET = $UPSTREAM@${BASE:0:7} + patches + overlay"
echo "Next: see $HERE/README.md (image, datasets, running)."
