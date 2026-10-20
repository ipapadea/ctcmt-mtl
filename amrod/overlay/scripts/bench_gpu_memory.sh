#!/usr/bin/env bash
# Peak-GPU-memory benchmark for each headline method.
# For each config, runs eval on ACDC fog only (~1 min per method) with
# nvidia-smi polling every 100ms in the background. Reports peak MB used.
set -euo pipefail
GPU=${1:-0}

# (Display name, config path)
CONFIGS=(
  "PFN_source_only|detectron2/configs/Cityscapes/panoptic_fpn_R_50.yaml"
  "CTCMT_MTL_baseline|detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_seed0_pfn_R_50_ACDC.yaml"
  "CTCMT_MTL_V2|detectron2/configs/Cityscapes/ctcmt_v2_seed0_pfn_R_50_ACDC.yaml"
  "CTCMT_MTL_V2_CTPV|detectron2/configs/Cityscapes/ctcmt_e1_ctpv_seed0_pfn_R_50_ACDC.yaml"
  "CTCMT_Det_MR|detectron2/configs/Cityscapes/ctcmt_det_mr_seed0_R_50_ACDC.yaml"
)

MEM_LOG=/tmp/gpu_mem_bench.log
: > "${MEM_LOG}"
echo "method,peak_mib" > "${MEM_LOG}"

for entry in "${CONFIGS[@]}"; do
  NAME="${entry%|*}"
  CFG="${entry#*|}"
  echo "=== ${NAME} ==="

  # Poll nvidia-smi every 100ms and keep the max.
  MEM_TMP=$(mktemp)
  (
    while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" \
        2>/dev/null | tr -d ' ' >> "${MEM_TMP}"
      echo "" >> "${MEM_TMP}"
      sleep 0.1
    done
  ) &
  POLL_PID=$!

  # Which ACDC test we use depends on the config type.
  if [[ "${NAME}" == "CTCMT_Det_MR" ]]; then
    TEST_DS='("acdc_fog",)'
  else
    TEST_DS='("acdc_fog_mtl",)'
  fi

  # For source-only we still run through eval-only; adaptation loop doesn't trigger.
  timeout 180 docker run --rm --gpus "\"device=${GPU}\"" --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -v /home/ilias/AMROD:/workspace/amrod \
    -v /data/vgcmt/datasets/cityscapes:/data/vgcmt/datasets/cityscapes:ro \
    -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro \
    -v /data/ilias/panoptic_fpn/output/coco_annotations:/datasets/annotations:ro \
    -v /data/ilias/acdc:/datasets/ACDC:ro \
    -v /data/vgcmt/datasets/cityscapes_foggy:/datasets/cityscapes_foggy:ro \
    -v /data/ilias/panoptic_fpn/output:/workspace/output \
    -w /workspace/amrod \
    -e DETECTRON2_DATASETS=/datasets \
    -e PYTHONPATH=/workspace/amrod/detectron2 \
    amrod:latest bash -c "
      export HOME=/tmp
      pip install --quiet --user --no-warn-script-location shapely
      python -u detectron2/tools/train_net.py \
        --config-file ${CFG} \
        --eval-only --num-gpus 1 \
        DATASETS.TEST '${TEST_DS}' \
        OUTPUT_DIR /workspace/output/bench_${NAME}
    " > /tmp/bench_${NAME}.out 2>&1 || true

  kill $POLL_PID 2>/dev/null || true
  wait $POLL_PID 2>/dev/null || true

  PEAK=$(sort -n "${MEM_TMP}" | grep -v '^$' | tail -1)
  echo "${NAME},${PEAK}" >> "${MEM_LOG}"
  echo "${NAME}: peak = ${PEAK} MiB"
  rm -f "${MEM_TMP}"
done

echo ""
echo "=== SUMMARY (also saved to ${MEM_LOG}) ==="
cat "${MEM_LOG}"
