#!/usr/bin/env bash
# Cityscapes-to-Cityscapes-C CTTA batch (Table 2 style).
#
# Runs 6 methods sequentially on a single GPU (default: 0).
# Each method shares the same 12-corruption test stream.
#
#   amrod_cs_c          AMROD det-only, our MkRCNN source
#   cotta_cs_c          CoTTA seg-only
#   ctcmt_det_cs_c      CT-CMT det-only, PFN source
#   ctcmt_seg_cs_c      CT-CMT seg-only, PFN source
#   ctcmt_mtl_cs_c      CT-CMT MTL v4b (ours, full method)
#   ctcmt_det_mr_cs_c   CT-CMT det-only, MkRCNN source (fair vs AMROD)
#
# Usage:
#   bash scripts/run_cityscapes_c_batch.sh [GPU]   (default GPU=0)
#
# Output dirs (inside container → /workspace/output):
#   /data/ilias/panoptic_fpn/output/ctta_cs_c/<method>/
#
# Logs: /tmp/cs_c_<method>.log

set -euo pipefail
GPU="${1:-0}"
RUNNER="bash /home/ilias/AMROD/scripts/run_ctta_acdc.sh"
LOGS=/tmp

echo "=== Cityscapes-C CTTA batch (GPU ${GPU}) ==="
echo "Dataset root: /data/vgcmt/datasets/cityscapes_c_amrod"
echo "Methods: amrod | cotta | ctcmt_det | ctcmt_seg | ctcmt_mtl"
echo ""

for TRACK in amrod_cs_c cotta_cs_c ctcmt_det_cs_c ctcmt_seg_cs_c ctcmt_mtl_cs_c ctcmt_det_mr_cs_c; do
    echo "--- [$(date +%H:%M)] START ${TRACK} ---"
    $RUNNER "$GPU" "$TRACK" \
        2>&1 | tee "${LOGS}/cs_c_${TRACK}.log"
    echo "--- [$(date +%H:%M)] DONE  ${TRACK} ---"
    echo ""
done

echo "=== All done. Results summary ==="
python3 - <<'EOF'
import re, glob

CORRUPTIONS = ["defocus_blur","glass_blur","motion_blur","zoom_blur",
               "snow","frost","fog","brightness","contrast",
               "elastic_transform","pixelate","jpeg_compression"]

import os
for track in ["amrod_cs_c","cotta_cs_c","ctcmt_det_cs_c","ctcmt_seg_cs_c","ctcmt_mtl_cs_c","ctcmt_det_mr_cs_c"]:
    logf = f"/tmp/cs_c_{track}.log"
    if not os.path.exists(logf):
        continue
    log = open(logf).read()
    vals = {}
    for c in CORRUPTIONS:
        m = re.search(
            rf"Evaluation results for {c}[^\n]*\n.*?copypaste: ([0-9.,]+)",
            log, re.DOTALL
        )
        if m:
            vals[c] = float(m.group(1).split(",")[1])
    if vals:
        mean = sum(vals.values()) / len(vals)
        row = "  ".join(f"{vals.get(c, float('nan')):.1f}" for c in CORRUPTIONS)
        print(f"{track:<20}  {row}  Mean={mean:.1f}")
    else:
        print(f"{track:<20}  (no results)")
EOF
