#!/usr/bin/env bash
set -uo pipefail

SEED=""
GPU=""

usage() {
  echo "Usage: $0 --seed {0|42|123} --gpu GPU"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$SEED" in
  0|42|123) ;;
  *)
    echo "ERROR: --seed must be 0, 42 or 123" >&2
    exit 1
    ;;
esac

[[ -n "$GPU" ]] || {
  echo "ERROR: --gpu required" >&2
  exit 1
}

ROOT="/data/ilias/panoptic_fpn/output/ft_historical"
STATE="${ROOT}/_queue_state"
LOG_DIR="${STATE}/logs"

mkdir -p "$LOG_DIR"

METHODS=(
  ctcmt-det-ft
  ctcmt-seg-ft
  ctcmt-mtl-v2-ft
)

BENCHES=(
  acdc-lt
  cs-c
  cs-c-lt
)

TOTAL=9
INDEX=0
FAILURES=0

echo "============================================================"
echo "FT HISTORICAL QUEUE"
echo "SEED : $SEED"
echo "GPU  : $GPU"
echo "JOBS : $TOTAL"
echo "============================================================"

for METHOD in "${METHODS[@]}"; do
  for BENCH in "${BENCHES[@]}"; do
    INDEX=$((INDEX + 1))

    RUN_NAME="${METHOD}__${BENCH}__seed${SEED}"
    DONE="${STATE}/${RUN_NAME}.done"
    FAILED="${STATE}/${RUN_NAME}.failed"
    OUT="${ROOT}/${RUN_NAME}"
    LOG="${LOG_DIR}/${RUN_NAME}.log"

    echo
    echo "============================================================"
    echo "[$INDEX/$TOTAL] $RUN_NAME"
    echo "============================================================"

    if [[ -f "$DONE" ]]; then
      echo "SKIP: already completed"
      continue
    fi

    # Safety: never overwrite a partially existing experiment.
    if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
      echo "BLOCKED: output exists but no .done marker:"
      echo "  $OUT"
      echo "Manual inspection required."
      echo "$(date -Is) BLOCKED existing output" > "$FAILED"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    rm -f "$FAILED"

    echo "LOG: $LOG"
    echo "START: $(date -Is)" | tee "$LOG"

    set +e
    ./scripts/run_ft_historical.sh \
      --method "$METHOD" \
      --bench "$BENCH" \
      --seed "$SEED" \
      --gpu "$GPU" \
      2>&1 | tee -a "$LOG"

    RC=${PIPESTATUS[0]}
    set -e

    if [[ "$RC" -eq 0 ]]; then
      echo "DONE: $(date -Is)" | tee -a "$LOG"
      echo "$(date -Is)" > "$DONE"
      rm -f "$FAILED"
    else
      echo "FAILED rc=$RC: $(date -Is)" | tee -a "$LOG"
      echo "$(date -Is) rc=$RC" > "$FAILED"
      FAILURES=$((FAILURES + 1))
    fi
  done
done

echo
echo "============================================================"
echo "QUEUE FINISHED"
echo "SEED     : $SEED"
echo "GPU      : $GPU"
echo "FAILURES : $FAILURES"
echo "============================================================"

if [[ "$FAILURES" -gt 0 ]]; then
  exit 1
fi

exit 0
