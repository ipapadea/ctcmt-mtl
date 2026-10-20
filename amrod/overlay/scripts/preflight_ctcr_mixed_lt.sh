#!/usr/bin/env bash
set -euo pipefail

HOST_REPO="${HOST_REPO:-/home/ilias/AMROD}"
CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/ilias/cityscapes_pfn}"
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT:-/data/vgcmt/datasets/cityscapes/annotations}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"

echo "Checking host paths..."
for d in \
  "${HOST_REPO}" \
  "${CSC_ROOT}/fog" \
  "${CSC_ROOT}/motion_blur" \
  "${CSC_ROOT}/snow" \
  "${CSC_ROOT}/brightness" \
  "${CSC_ROOT}/defocus_blur" \
  "${CITYSCAPES_ROOT}/gtFine/val" \
  "${CITYSCAPES_ANN_ROOT}"
do
  printf "%-75s " "${d}"
  [[ -e "${d}" ]] && echo "OK" || { echo "MISSING"; exit 2; }
done

printf "%-75s " "${CITYSCAPES_ANN_ROOT}/instancesonly_filtered_gtFine_val.json"
[[ -f "${CITYSCAPES_ANN_ROOT}/instancesonly_filtered_gtFine_val.json" ]] \
  && echo "OK" || { echo "MISSING"; exit 2; }

echo
echo "Checking mounted paths inside Docker..."
docker run --rm \
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
  -v "${CITYSCAPES_ANN_ROOT}:/datasets/cityscapes/annotations:ro" \
  -v "${CSC_ROOT}/fog:/datasets/fog:ro" \
  -v "${CSC_ROOT}/motion_blur:/datasets/motion_blur:ro" \
  -v "${CSC_ROOT}/snow:/datasets/snow:ro" \
  -v "${CSC_ROOT}/brightness:/datasets/brightness:ro" \
  -v "${CSC_ROOT}/defocus_blur:/datasets/defocus_blur:ro" \
  "${DOCKER_IMAGE}" bash -c '
    set -e
    test -f /datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json
    test -d /datasets/cityscapes/gtFine/val
    for c in fog motion_blur snow brightness defocus_blur; do
      test -d "/datasets/$c"
      echo "$c: OK"
    done
    echo "semantic GT examples:"
    find /datasets/cityscapes/gtFine/val -name "*_gtFine_labelTrainIds.png" | head -3
  '

echo
echo "Preflight OK."
