#!/usr/bin/env bash
# Live dashboard for CTTA runs. Works on either machine.
#
#   watch -n 30 -c bash scripts/monitor.sh
#   LOGS=/data/ilias/panoptic_fpn/output/logs watch -n 30 -c bash scripts/monitor.sh
#
# ACTIVE_MIN controls how recently a log must have been written to count as live.

LOGS="${LOGS:-/media/ilias/DATA/ilias/amrod_output/logs}"
ACTIVE_MIN="${ACTIVE_MIN:-15}"
now=$(date +%s)

expected_for() {   # infer the eval count a complete run must reach
  case "$1" in
    *acdc_lt*|*acdcLT*) echo 40 ;;
    *csc_lt*|*cscLT*)   echo 50 ;;
    *csc12*|*cs_c*) echo 12 ;;
    cd ~
    tar czf ~/amrod_gpu1_snapshot_$(date +%F_%H%M).tgz \
        --exclude='.git' --exclude='__pycache__' AMROD
    ls -lh ~/amrod_gpu1_snapshot_*.tgz    *acdc*)             echo  4 ;;
    *)                  echo  0 ;;
  esac
}

printf '\033[1m%s   %s\033[0m\n' "$(hostname)" "$(date '+%F %H:%M:%S')"

echo
echo "GPU  mem_used   util"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
  | awk -F', ' '{printf "  %s  %9s  %5s\n", $1, $2, $3}'

echo
n_ctr=$(docker ps -q --filter ancestor=amrod:latest 2>/dev/null | wc -l)
echo "containers: ${n_ctr}    shell wrappers:"
pgrep -af 'run_remote_baselines|run_mixed_lt_local|run_ctcr_acdc_local|run_acdc_lt_baselines' 2>/dev/null \
  | sed 's/^/  /' | cut -c1-120 | head -6
[[ ${n_ctr} -eq 0 ]] && echo "  (none)"

echo
printf '%-34s %8s %10s %9s  %s\n' "ACTIVE RUN" "evals" "progress" "updated" "state"
printf '%.0s-' {1..96}; echo
found=0
for f in "${LOGS}"/*.log; do
  [[ -e "$f" ]] || continue
  age=$(( now - $(stat -c %Y "$f") ))
  (( age > ACTIVE_MIN * 60 )) && continue
  found=1
  name=$(basename "$f" .log)
  exp=$(expected_for "$name")
  ev=$(grep -c 'in csv format' "$f")
  prog=$(grep -o 'Inference done [0-9]*/[0-9]*' "$f" | tail -1 | awk '{print $3}')
  state="running"
  (( exp > 0 && ev >= exp )) && state="DONE"
  grep -q 'Traceback (most recent call last)' "$f" && state="FAILED"
  printf '%-34s %4s/%-3s %10s %8ss  %s\n' \
    "${name:0:34}" "$ev" "${exp:-?}" "${prog:-—}" "$age" "$state"
done
(( found == 0 )) && echo "  (no log touched in the last ${ACTIVE_MIN} min)"

echo
echo "recently finished:"
for f in "${LOGS}"/*.log; do
  [[ -e "$f" ]] || continue
  name=$(basename "$f" .log); exp=$(expected_for "$name")
  (( exp == 0 )) && continue
  ev=$(grep -c 'in csv format' "$f")
  (( ev >= exp )) && printf '  %-34s %s/%s\n' "$name" "$ev" "$exp"
done | tail -8
