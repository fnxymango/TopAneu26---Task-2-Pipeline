#!/usr/bin/env bash
# V2-0 큐 — 돌고 있는 neckflip.py · m1census.py 가 끝나면 ICA 분할 측정을 시작한다.
# 프로세스 확인은 argv 를 **정확 대조**한다. pgrep·부분문자열 매칭 금지(오늘 두 번 사고).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee/nnunet/scripts:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/v20.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v20] $*"; }

busy(){
  local n=0 p a2 a3
  for p in /proc/[0-9]*; do
    [ -r "$p/cmdline" ] || continue
    a2=$(tr '\0' '\n' < "$p/cmdline" | sed -n 2p)
    a3=$(tr '\0' '\n' < "$p/cmdline" | sed -n 3p)
    for f in "$a2" "$a3"; do
      case "$f" in "$D/neckflip.py"|"$D/m1census.py") n=$((n+1));; esac
    done
  done
  echo $n
}

log "대기 시작 — neckflip.py · m1census.py 종료를 기다린다"
w=0
while [ "$(busy)" -gt 0 ]; do sleep 30; w=$((w+30)); [ $((w%300)) -eq 0 ] && log "  대기 ${w}s · 워커 $(busy)"; done
log "선행 작업 종료 확인 · V2-0 시작"

"$PY" -u "$D/v20_icasplit.py" > "$D/v20_measure.log" 2>&1 || { log "★측정 실패 — v20_measure.log"; exit 1; }
n=$("$PY" -c "import json;print(len(json.load(open('$D/v20_icasplit.json'))))" 2>/dev/null || echo 0)
log "측정 완료 side $n"
"$PY" "$D/v20_report.py" > "$R/experiments/V1_vessel_axis/RESULTS_V20.md" 2>&1
log "V2-0 완료 → experiments/V1_vessel_axis/RESULTS_V20.md"
touch "$D/.done_v20"
