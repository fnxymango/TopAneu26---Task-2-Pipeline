#!/usr/bin/env bash
# 실험 감시자 (사용자 지시 2026-08-13: "문제생기면 알아서대처해 / 안끊기게").
# 5분마다 점검하고 스스로 복구한다:
#   1) 체인이 죽었는데 완료표시가 없으면 -> 멱등 재개 스크립트로 되살림(--c 로 checkpoint 이어받기)
#   2) 디스크 여유가 부족하면 -> 안전한 임시물(폐기 실험/nohup 잔재) 정리 후 경고
#   3) GPU에 우리 프로세스가 없는데 체인은 살아있는 유령 상태 감지
# 자기 자신은 nohup+disown으로 세션과 분리해 띄운다.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
E="$TOPANEU_ROOT/experiments"
LOG="$E/supervisor.log"
export TOPANEU_ROOT
log() { echo "[supervisor $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

INTERVAL=300
MIN_FREE_GB=15

log "감시 시작 (주기 ${INTERVAL}s, 최소여유 ${MIN_FREE_GB}GB)"

alive() { pgrep -f "$1" >/dev/null 2>&1; }

check_chain() {  # $1=라벨 $2=체인로그 $3=완료문구 $4=재개스크립트 $5=원본체인패턴
  local name="$1" clog="$2" marker="$3" resume="$4" pat="$5"
  if grep -q "$marker" "$clog" 2>/dev/null; then return; fi          # 이미 완료
  if alive "$pat" || alive "$(basename "$resume")"; then return; fi   # 아직 살아있음
  log "⚠️ $name 체인이 완료표시 없이 종료됨 -> 재개 스크립트 기동"
  TOPANEU_ROOT="$TOPANEU_ROOT" nohup bash "$resume" >/dev/null 2>&1 &
  disown
  log "   $name 재개 기동 완료"
}

while true; do
  # ---- 디스크 ----
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "${FREE:-999}" -lt "$MIN_FREE_GB" ]; then
    log "⚠️ 디스크 여유 ${FREE}GB (< ${MIN_FREE_GB}GB) — 안전 임시물 정리"
    rm -rf "$E"/_discarded_* 2>/dev/null
    rm -f  "$E"/nohup_*.out 2>/dev/null
    rm -rf "$E"/__t__* 2>/dev/null
    FREE2=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
    log "   정리 후 ${FREE2}GB"
    [ "${FREE2:-999}" -lt 8 ] && log "🚨 여전히 부족(${FREE2}GB) — 학습이 곧 실패할 수 있음. 수동 개입 필요"
  fi

  # ---- 체인 생존 ----
  check_chain "GPU0" "$E/gpu0_chain_a6.log" "GPU0 체인 완료" \
              "$SCRIPTS/chain_gpu0_resume.sh" "run_gpu0_chain_a6.sh"
  check_chain "GPU1" "$E/gpu1_chain_c1.log" "GPU1 체인 완료" \
              "$SCRIPTS/chain_gpu1_resume.sh" "run_gpu1_chain_c1.sh"

  sleep "$INTERVAL"
done
