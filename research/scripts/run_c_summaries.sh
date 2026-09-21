#!/usr/bin/env bash
# C계열 실험 폴더/summary 자동 생성·갱신 러너 (2026-08-17).
#
# 세션과 무관하게 살아남아야 하므로 nohup + disown 으로 띄운다.
# C27/C28 이 아직 돌고 있어 결과 json 이 계속 늘어난다 -> 주기적으로 재생성하고,
# **산출물이 조용해지면** 마지막으로 한 번 더 돌고 종료한다.
# 대기 판정에 pgrep 을 쓰지 않는다 (PROJECT_RULES.md §4 — 앞서 두 번 정체시킨 원인).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
[ -x "$PY" ] || PY=/usr/bin/python3
LOG="$E/c_summaries.log"
QUIET=1200      # 20분간 새 산출물/로그 갱신 없으면 종료
PERIOD=300      # 5분 주기
MAX=48          # 최대 4시간

log(){ echo "[csum $(date -u +'%m-%d %H:%M:%S')] $*" >> "$LOG"; }

newest_mtime(){  # 감시 대상들 중 가장 최근 수정시각(epoch)
  local newest=0 t
  for f in "$A"/c5_eval_*.json "$E"/c27_chain.log "$E"/c28_chain.log; do
    [ -f "$f" ] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$t" -gt "$newest" ] && newest=$t
  done
  echo "$newest"
}

cd "$S" || exit 1
export TOPANEU_ROOT="$R"
log "=========================================================="
log "시작 (주기 ${PERIOD}s · 정체판정 ${QUIET}s · 최대 $((MAX*PERIOD/60))분)"

for i in $(seq 1 "$MAX"); do
  log "생성 #$i"
  "$PY" -u organize_c_experiments_v2.py >> "$LOG" 2>&1 || log "  생성 실패 status=$?"
  NOW=$(date +%s); NEW=$(newest_mtime); AGE=$(( NOW - NEW ))
  log "  최신 산출물 ${AGE}s 전"
  if [ "$AGE" -ge "$QUIET" ]; then
    log "실험이 조용해짐 (${AGE}s) — 최종 생성 후 종료"
    "$PY" -u organize_c_experiments_v2.py >> "$LOG" 2>&1
    log "완료"
    exit 0
  fi
  sleep "$PERIOD"
done
log "최대 반복 도달 — 최종 생성 후 종료"
"$PY" -u organize_c_experiments_v2.py >> "$LOG" 2>&1
log "완료"
