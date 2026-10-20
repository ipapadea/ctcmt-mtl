#!/usr/bin/env bash
set -euo pipefail
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
printf '%-46s %-18s %9s %6s %s\n' EXP STATUS EVALS ERRORS LAST_LINE
bad=0
for meta in "$HOST_OUT"/paper26_meta/paper26_*.txt; do
  [[ -e "$meta" ]] || continue
  name="$(basename "$meta" .txt)"; log="$HOST_OUT/logs/$name.log"
  status="$(sed -n 's/^status=//p' "$meta" | tail -n 1)"
  want="$(sed -n 's/^expected_evals=//p' "$meta" | tail -n 1)"
  evals=0; errors=0; last=""
  if [[ -f "$log" ]]; then
    evals="$(grep -c 'in csv format' "$log" || true)"
    errors="$(grep -Ec 'Traceback|CUDA out of memory|^RuntimeError:' "$log" || true)"
    last="$(tail -n 1 "$log" | tr -s ' ' | cut -c1-60)"
  fi
  [[ -n "$status" ]] || status="RUNNING/KILLED"   # no end record: still running or batch died
  [[ "$status" == COMPLETE ]] || bad=1
  printf '%-46s %-18s %9s %6s %s\n' "$name" "$status" "${evals}/${want:-?}" "$errors" "$last"
done
exit "$bad"
