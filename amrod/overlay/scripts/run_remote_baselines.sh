#!/usr/bin/env bash
set -euo pipefail

# Same-source baseline suite for the remote machine.
#
# Every run here evaluates a COMPETING method on OUR Panoptic-FPN MTL source
# checkpoint. These results are independent of how our own method evolves, so
# they can never be invalidated by later changes to the base configuration --
# which is exactly what makes them safe to run remotely and unattended.
#
#   bash scripts/run_remote_baselines.sh list
#   bash scripts/run_remote_baselines.sh acdc_lt GPU [SEED]   # AMROD Table 5
#   bash scripts/run_remote_baselines.sh csc_lt  GPU [SEED]   # AMROD Table 3
#   bash scripts/run_remote_baselines.sh csc_12  GPU [SEED]   # AMROD Table 2
#
# Paths default to the remote layout; override any of them via env:
#   HOST_REPO HOST_OUT CSC_ROOT ACDC_ROOT CITYSCAPES_ROOT CITYSCAPES_ANN_ROOT

STAGE="${1:-list}"
GPU="${2:-0}"
SEED="${3:-0}"
# Optional 4th arg: run only one method (amrod|cotta|tent). Default: all three.
ONLY="${4:-all}"

HOST_REPO="${HOST_REPO:-/home/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/data/ilias/panoptic_fpn/output}"
CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
ACDC_ROOT="${ACDC_ROOT:-/data/ilias/acdc}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/vgcmt/datasets/cityscapes}"
# Set to "" for segmentation-only arms (CoTTA/TENT): they need no detection json,
# and bind-mounting onto an `annotations` symlink inside the parent mount fails.
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT-${CITYSCAPES_ROOT}/annotations}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
CFG=detectron2/configs/Cityscapes

CSC_12=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog
        brightness contrast elastic_transform pixelate jpeg_compression)
CSC_LT_CYCLE=(fog motion_blur snow brightness defocus_blur)
ACDC_CYCLE=(fog night rain snow)

join_stream() {  # join_stream SUFFIX REPEATS ITEMS...
  local suffix="$1"; shift; local reps="$1"; shift
  local out="(" i c
  for ((i = 0; i < reps; i++)); do for c in "$@"; do out+="\"${c}${suffix}\","; done; done
  echo "${out%,})"
}

run() {  # run EXP CONFIG STREAM MOUNT_KIND METHOD
  local exp="$1" cfg="$2" stream="$3" kind="$4" method="${5:-}"
  if [[ "${ONLY}" != "all" && -n "${method}" && "${ONLY}" != "${method}" ]]; then
    echo "--- skip ${exp} (ONLY=${ONLY})"; return 0
  fi
  local out="/workspace/output/ctta_acdc/${exp}"
  local log="${HOST_OUT}/logs/${exp}.log"
  # Preflight must never clobber a completed run's log.
  if [[ "${PREFLIGHT:-0}" == "1" ]]; then
    log="${HOST_OUT}/logs/${exp}.preflight.log"
  fi
  mkdir -p "$(dirname "${log}")"

  local mounts=()
  if [[ "${kind}" == "csc" ]]; then
    # Mount every corruption dir; harmless if the stream uses only a subset.
    local c
    for c in "${CSC_12[@]}"; do
      if [[ -d "${CSC_ROOT}/${c}" ]]; then
        mounts+=(-v "${CSC_ROOT}/${c}:/datasets/${c}:ro")
      fi
    done
    if [[ ${#mounts[@]} -eq 0 ]]; then
      echo "ERROR: no corruption dirs found under ${CSC_ROOT}" >&2; exit 2
    fi
  else
    [[ -d "${ACDC_ROOT}" ]] || { echo "ERROR: missing ${ACDC_ROOT}" >&2; exit 2; }
    mounts+=(-v "${ACDC_ROOT}:/datasets/ACDC:ro")
  fi

  echo "=== ${exp}  (GPU ${GPU}, seed ${SEED})"
  # PREFLIGHT=1 validates every dataset in the stream in ~30 s instead of
  # discovering a registration or missing-label problem hours into a run.
  local payload
  if [[ "${PREFLIGHT:-0}" == "1" ]]; then
    payload="python - '${stream}' <<'PYEOF'
import sys
from detectron2.data import DatasetCatalog
names = [n for n in sys.argv[1].strip('()').replace('\"','').split(',') if n]
bad = []
for n in dict.fromkeys(names):
    try:
        d = DatasetCatalog.get(n)
        print(f'  OK   {n:30s} {len(d)} records', flush=True)
    except Exception as e:
        print(f'  FAIL {n:30s} {type(e).__name__}: {str(e)[:90]}', flush=True)
        bad.append(n)
sys.exit(1 if bad else 0)
PYEOF"
  else
    payload="python detectron2/tools/train_net.py --config-file ${cfg} \
        --eval-only --num-gpus 1 \
        OUTPUT_DIR ${out} SEED ${SEED} DATASETS.TEST '${stream}'"
    rm -rf "${HOST_OUT}/ctta_acdc/${exp}"
  fi
  local ann_mount=()
  if [[ -n "${CITYSCAPES_ANN_ROOT}" ]]; then
    ann_mount=(-v "${CITYSCAPES_ANN_ROOT}:/datasets/cityscapes/annotations:ro")
  fi

  # Never abort the whole batch on one failed arm; record and continue.
  set +e
  docker run --rm --gpus "\"device=${GPU}\"" --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e NVIDIA_TF32_OVERRIDE=0 \
    -e DETECTRON2_DATASETS=/datasets -e PYTHONPATH=/workspace/amrod/detectron2 \
    -v "${HOST_REPO}:/workspace/amrod:ro" \
    -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
    "${ann_mount[@]}" \
    "${mounts[@]}" \
    -v "${HOST_OUT}:/workspace/output" \
    -w /workspace/amrod \
    "${DOCKER_IMAGE}" bash -c "${payload}" 2>&1 | tee "${log}" | { [[ "${PREFLIGHT:-0}" == "1" ]] && grep -E "^  (OK|FAIL)" || cat > /dev/null; }
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${PREFLIGHT:-0}" == "1" ]]; then
    [[ ${rc} -eq 0 ]] && echo "  --> ${exp} PREFLIGHT PASS" || { echo "  --> ${exp} PREFLIGHT FAIL"; tail -5 "${log}"; }
  elif [[ ${rc} -ne 0 ]]; then
    echo "!!! ${exp} FAILED (rc=${rc}); last lines:"
    tail -25 "${log}"
  else
    echo "--- ${exp} OK ($(grep -c 'in csv format' "${log}") evals)"
  fi
}

case "${STAGE}" in
list)
  cat <<'TXT'
All arms below run a COMPETING method on OUR Panoptic-FPN checkpoint.
They are method-independent, so they will not need re-running.

  acdc_lt  (~3.5 h each)  AMROD Table 5 replica, fog->night->rain->snow x10
             amrod_pfnsrc_acdc_lt   cotta_pfnsrc_acdc_lt   tent_pfnsrc_acdc_lt
             >>> HIGHEST VALUE: this is the table we already win, and it is the
                 only one whose baselines are still cross-source.

  csc_lt   (~5.5 h each)  AMROD Table 3 replica, 5 corruptions x10
             amrod_pfnsrc_csc_lt    cotta_pfnsrc_csc_lt    tent_pfnsrc_csc_lt

  csc_12   (~1.5 h each)  AMROD Table 2 replica, 12 corruptions once
             amrod_pfnsrc_csc12     cotta_pfnsrc_csc12     tent_pfnsrc_csc12

Ours (already measured, for reference):
  ACDC x10   41.303 +/- 0.789 AP50 | 37.628 +/- 1.606 mIoU   (source 32.97 / 30.44)
  CS-C LT    ~18.7 AP50 | ~29.1 mIoU                          (source 13.17 / 27.21)
  CS-C 12    14.67 AP50 | 27.57 mIoU                          (source 12.70 / 26.25)
TXT
  ;;
acdc_lt)
  S=$(join_stream ""        10 "${ACDC_CYCLE[@]/#/acdc_}")
  SS=$(join_stream "_semseg" 10 "${ACDC_CYCLE[@]/#/acdc_}")
  run "amrod_pfnsrc_acdc_lt_s${SEED}" "${CFG}/amrod_pfn_R_50_ACDC.yaml"         "${S}"  acdc amrod
  run "cotta_pfnsrc_acdc_lt_s${SEED}" "${CFG}/cotta_semseg_pfn_R_50_ACDC.yaml"  "${SS}" acdc cotta
  run "tent_pfnsrc_acdc_lt_s${SEED}"  "${CFG}/tent_semseg_pfn_R_50_ACDC.yaml"   "${SS}" acdc tent
  ;;
csc_lt)
  S=$(join_stream ""         10 "${CSC_LT_CYCLE[@]}")
  SS=$(join_stream "_semseg" 10 "${CSC_LT_CYCLE[@]}")
  run "amrod_pfnsrc_csc_lt_s${SEED}" "${CFG}/amrod_pfn_R_50_CS_C.yaml"          "${S}"  csc amrod
  run "cotta_pfnsrc_csc_lt_s${SEED}" "${CFG}/cotta_semseg_pfn_R_50_CS_C.yaml"   "${SS}" csc cotta
  run "tent_pfnsrc_csc_lt_s${SEED}"  "${CFG}/tent_semseg_pfn_R_50_CS_C.yaml"    "${SS}" csc tent
  ;;
csc_12)
  S=$(join_stream ""         1 "${CSC_12[@]}")
  SS=$(join_stream "_semseg" 1 "${CSC_12[@]}")
  run "amrod_pfnsrc_csc12_s${SEED}" "${CFG}/amrod_pfn_R_50_CS_C.yaml"           "${S}"  csc amrod
  run "cotta_pfnsrc_csc12_s${SEED}" "${CFG}/cotta_semseg_pfn_R_50_CS_C.yaml"    "${SS}" csc cotta
  run "tent_pfnsrc_csc12_s${SEED}"  "${CFG}/tent_semseg_pfn_R_50_CS_C.yaml"     "${SS}" csc tent
  ;;
*)
  echo "unknown stage '${STAGE}' -- run: bash $0 list" >&2; exit 2 ;;
esac
