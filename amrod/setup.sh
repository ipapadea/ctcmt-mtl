#!/usr/bin/env bash
# Build the CTCMT / AMROD working tree: upstream AMROD at a pinned commit + our patch + our files.
#
#   bash setup.sh TARGET_DIR
#
# The upstream repository has no license file, so this repository does not redistribute it: the six upstream
# files we changed are shipped as a patch (patches/amrod_upstream_changes.patch) and everything else as new files
# (overlay/).
set -euo pipefail
TARGET="${1:?usage: bash setup.sh TARGET_DIR}"
UPSTREAM="${UPSTREAM:-https://github.com/ShileiCao/AMROD.git}"
BASE="${BASE:-199133286d603159c6bcae2ce4338dbc31fb50fd}"   # last upstream commit our work is based on
HERE="$(cd "$(dirname "$0")" && pwd)"

[[ -e "$TARGET" ]] && { echo "ERROR: $TARGET already exists" >&2; exit 2; }
mkdir -p "$TARGET" && cd "$TARGET"
git init -q . && git remote add origin "$UPSTREAM"
git fetch -q --depth 1 origin "$BASE" && git checkout -q FETCH_HEAD
git apply --whitespace=nowarn "$HERE/patches/amrod_upstream_changes.patch"
cp -a "$HERE/overlay/." .
echo "OK: $TARGET = $UPSTREAM@${BASE:0:7} + patches + overlay"
echo "Next: see $HERE/README.md (environment, datasets, running)."
