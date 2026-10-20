#!/usr/bin/env bash
# Build a PDF from a Markdown file with a Unicode-safe font stack.
# Usage:  bash scripts/build_md_pdf.sh <input.md> [output.pdf]
set -euo pipefail

IN="${1:?usage: build_md_pdf.sh <input.md> [output.pdf]}"
OUT="${2:-${IN%.md}.pdf}"

pandoc "${IN}" \
  -o "${OUT}" \
  --pdf-engine=xelatex \
  -V mainfont="DejaVu Serif" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=1in \
  -V colorlinks=true

echo "Built ${OUT} ($(stat -c %s "${OUT}") bytes)"
