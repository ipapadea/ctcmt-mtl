#!/usr/bin/env bash
set -euo pipefail

METHOD=""
BENCH=""
SEED=""
GPU="0"
DRY_RUN=false

usage() {
  cat <<USAGE
Usage:
  $0 --method METHOD --bench BENCH --seed SEED [--gpu GPU] [--dry-run]

Methods:
  ctcmt-det-ft
  ctcmt-seg-ft
  ctcmt-mtl-v2-ft

Benchmarks:
  acdc-lt
  cs-c
  cs-c-lt

Seeds:
  0, 42, 123
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"; shift 2 ;;
    --bench)
      BENCH="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --gpu)
      GPU="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

[[ -n "$METHOD" ]] || { echo "ERROR: --method required"; exit 1; }
[[ -n "$BENCH"  ]] || { echo "ERROR: --bench required"; exit 1; }
[[ -n "$SEED"   ]] || { echo "ERROR: --seed required"; exit 1; }

case "$SEED" in
  0|42|123) ;;
  *)
    echo "ERROR: seed must be one of {0,42,123}" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Exact FINAL_TABLES historical method configurations.
# We do NOT modify these YAMLs. Only SEED and DATASETS.TEST are overridden.
# ---------------------------------------------------------------------------
case "$METHOD" in
  ctcmt-det-ft)
    CFG="detectron2/configs/Cityscapes/ctcmt_det_mr_R_50_ACDC.yaml"
    SUFFIX=""
    ;;

  ctcmt-seg-ft)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_small_aug_R_50_ACDC.yaml"
    SUFFIX="_semseg"
    ;;

  ctcmt-mtl-v2-ft)
    CFG="detectron2/configs/Cityscapes/ctcmt_v2_cross_fisher_pfn_R_50_ACDC.yaml"
    SUFFIX="_mtl"
    ;;

  *)
    echo "ERROR: unknown method '$METHOD'" >&2
    exit 1
    ;;
esac

make_tuple() {
  python3 - "$SUFFIX" "$@" <<'PY'
import sys

suffix = sys.argv[1]
names = sys.argv[2:]
items = [f'"{x}{suffix}"' for x in names]
print("(" + ",".join(items) + ")")
PY
}

repeat_cycle() {
  local repeats="$1"
  shift

  python3 - "$SUFFIX" "$repeats" "$@" <<'PY'
import sys

suffix = sys.argv[1]
repeats = int(sys.argv[2])
cycle = sys.argv[3:]

names = [
    f'"{name}{suffix}"'
    for _ in range(repeats)
    for name in cycle
]

print("(" + ",".join(names) + ")")
PY
}

case "$BENCH" in
  acdc-lt)
    TEST_DATASETS="$(
      repeat_cycle 10 \
        acdc_fog \
        acdc_night \
        acdc_rain \
        acdc_snow
    )"
    ;;

  cs-c)
    TEST_DATASETS="$(
      make_tuple \
        defocus_blur \
        glass_blur \
        motion_blur \
        zoom_blur \
        snow \
        frost \
        fog \
        brightness \
        contrast \
        elastic_transform \
        pixelate \
        jpeg_compression
    )"
    ;;

  cs-c-lt)
    TEST_DATASETS="$(
      repeat_cycle 10 \
        fog \
        motion_blur \
        snow \
        brightness \
        defocus_blur
    )"
    ;;

  *)
    echo "ERROR: unknown benchmark '$BENCH'" >&2
    exit 1
    ;;
esac

RUN_NAME="${METHOD}__${BENCH}__seed${SEED}"

OUT_CONTAINER="/workspace/output/ft_historical/${RUN_NAME}"
OUT_HOST="/data/ilias/panoptic_fpn/output/ft_historical/${RUN_NAME}"

echo "============================================================"
echo "FINAL_TABLES historical benchmark"
echo "METHOD : $METHOD"
echo "BENCH  : $BENCH"
echo "SEED   : $SEED"
echo "GPU    : $GPU"
echo "CFG    : $CFG"
echo "OUTPUT : $OUT_HOST"
echo "TEST   : $TEST_DATASETS"
echo "============================================================"

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

if [[ -d "$OUT_HOST" ]] && [[ -n "$(ls -A "$OUT_HOST" 2>/dev/null)" ]]; then
  echo "ERROR: output directory already exists and is non-empty:"
  echo "  $OUT_HOST"
  echo "Refusing to overwrite an existing experiment."
  exit 1
fi

mkdir -p "$OUT_HOST"

CS_C_ROOT="/data/vgcmt/datasets/cityscapes_c_amrod"

EXTRA_CS_C_MOUNTS=()
if [[ "$BENCH" == "cs-c" || "$BENCH" == "cs-c-lt" ]]; then
  for c in \
    defocus_blur \
    glass_blur \
    motion_blur \
    zoom_blur \
    snow \
    frost \
    fog \
    brightness \
    contrast \
    elastic_transform \
    pixelate \
    jpeg_compression
  do
    EXTRA_CS_C_MOUNTS+=(
      -v "${CS_C_ROOT}/${c}:/datasets/${c}:ro"
    )
  done
fi

docker run --rm \
  --gpus "\"device=${GPU}\"" \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -v /home/ilias/AMROD:/workspace/amrod \
  -v /data/vgcmt/datasets/cityscapes:/data/vgcmt/datasets/cityscapes:ro \
  -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro \
  -v /data/ilias/acdc:/datasets/ACDC:ro \
  -v /data/ilias/panoptic_fpn/output/coco_annotations:/datasets/annotations:ro \
  -v /data/ilias/panoptic_fpn/output:/workspace/output \
  "${EXTRA_CS_C_MOUNTS[@]}" \
  -w /workspace/amrod \
  -e DETECTRON2_DATASETS=/datasets \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  amrod:latest \
  bash -c "
    export HOME=/tmp

    echo '>>> START ${RUN_NAME} <<<'

    python detectron2/tools/train_net.py \
      --config-file '${CFG}' \
      --eval-only \
      --num-gpus 1 \
      SEED '${SEED}' \
      OUTPUT_DIR '${OUT_CONTAINER}' \
      DATASETS.TEST '${TEST_DATASETS}'

    echo '>>> DONE ${RUN_NAME} <<<'
  "
