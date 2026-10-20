# Shared helpers for paper26_batch_*.sh (sourced, not executed).
# Needs: GPU HOST_OUT DRY_RUN FORCE.

# GPU sanity + one-batch-per-GPU lock (fd 9 is inherited by docker, so an
# orphaned container keeps the lock). Real runs only.
p26_init() {
  [[ "$GPU" =~ ^[01]$ ]] || { echo "ERROR: GPU must be 0 or 1 (got '$GPU')" >&2; exit 2; }
  [[ "$DRY_RUN" == 1 ]] && return 0
  mkdir -p "$HOST_OUT/paper26_meta" "$HOST_OUT/logs"
  exec 9>"/tmp/amrod_gpu${GPU}.lock"
  flock -n 9 || { echo "ERROR: another paper26 batch already owns GPU $GPU" >&2; exit 4; }
  local used
  used="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 1500 )) && [[ "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    echo "ERROR: GPU $GPU already uses ${used} MiB (another job?). ALLOW_BUSY_GPU=1 to override." >&2; exit 4
  fi
}

# rc 0 = run this experiment, rc 1 = skip it. Never deletes anything:
# SKIP_COMPLETE=1 skips COMPLETE runs; FORCE=1 *moves* old artifacts to paper26_archive/.
p26_gate() {
  local exp="$1" log="$HOST_OUT/logs/$1.log" out="$HOST_OUT/ctta_acdc/$1" meta="$HOST_OUT/paper26_meta/$1.txt" p arc
  [[ -e "$log" || -e "$out" ]] || return 0
  if [[ "${SKIP_COMPLETE:-0}" == 1 ]] && grep -qx 'status=COMPLETE' "$meta" 2>/dev/null; then
    echo "  [SKIP] $exp already COMPLETE"; return 1
  fi
  if [[ "$FORCE" == 1 ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then echo "  [FORCE] would archive old artifacts of $exp"; return 0; fi
    arc="$HOST_OUT/paper26_archive/${exp}.$(date +%Y%m%dT%H%M%S)"; mkdir -p "$arc"
    for p in "$log" "$out" "$meta"; do [[ -e "$p" ]] && mv "$p" "$arc/"; done
    echo "  [FORCE] old artifacts of $exp archived -> $arc"; return 0
  fi
  echo "  [REFUSE] $exp already has a log/output dir; not overwriting (SKIP_COMPLETE=1 / FORCE=1)" >&2
  [[ "$DRY_RUN" == 1 ]] && return 1
  exit 3
}

# p26_finish EXP RC EXPECTED_EVALS LOG META  -> rc 0 only if COMPLETE
p26_finish() {
  local exp="$1" rc="$2" want="$3" log="$4" meta="$5" got=0 status
  [[ -f "$log" ]] && got="$(grep -c 'in csv format' "$log" || true)"
  if (( rc != 0 )); then status=FAILED
  elif (( got != want )); then status=INCOMPLETE
  else status=COMPLETE; fi
  { echo "end_time=$(date -Is)"; echo "exit_code=$rc"; echo "eval_count=$got"; echo "status=$status"; } >> "$meta"
  echo "[$status] $exp  evals=$got/$want  exit_code=$rc"
  [[ "$status" == COMPLETE ]]
}
