#!/usr/bin/env bash
# 큐 감시자 (2026-08-16). 사용자 지시: "좀 안 멈추게 잘 해봐".
#
# 왜 다시 만드나 — 지금까지 정체가 두 번 있었고 **둘 다 원인이 같다**:
#   `while pgrep -f '<스크립트명>'; do sleep; done` 형태의 대기가
#   내 셸 명령줄 wrapper(`bash -c '... <스크립트명> ...'`)까지 매칭해 영원히 안 풀렸다.
#   chain_best.sh 2.5시간, chain_after_c16.sh 3.4시간 손실.
#
# 그래서 이 감시자는 **pgrep을 쓰지 않는다.**
#   - 프로세스 생존 판정: /proc/<pid>/cmdline 의 argv를 직접 읽어
#     "bash <스크립트경로>" 형태(argv[1]이 스크립트)일 때만 인정한다.
#     `bash -c '...'` wrapper는 argv[1]이 "-c" 라서 절대 매칭되지 않는다.
#   - 진행 판정: 로그 파일 mtime. 프로세스가 살아 있어도 로그가 안 변하면 멈춘 것으로 본다.
#
# 복구: 큐를 재기동한다. 큐의 모든 단계는 산출물이 있으면 건너뛰므로 멱등이다.
# 중지: kill $(cat experiments/watchdog_queue.pid)
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
CHAIN="$S/chain_c44.sh"
QLOG="$E/c44_chain.log"
WLOG="$E/watchdog_queue.log"
DONE_MARK="=== 완료 ==="
STALL=2400          # 로그가 40분간 안 변하면 정체로 판단 (체인이 10분마다 하트비트를 찍음)
INTERVAL=180
echo $$ > "$E/watchdog_queue.pid"
exec >> "$WLOG" 2>&1
log(){ echo "[wq $(date -u +'%m-%d %H:%M:%S')] $*"; }

# argv[1] 이 스크립트 경로인 프로세스만 찾는다 (bash -c wrapper 는 argv[1]="-c" 라 제외됨)
chain_pid(){
  local p a0 a1
  for p in /proc/[0-9]*; do
    [ -r "$p/cmdline" ] || continue
    mapfile -d '' -t a < "$p/cmdline" 2>/dev/null || continue
    a0="${a[0]:-}"; a1="${a[1]:-}"
    case "$a0" in *bash|*sh) ;; *) continue;; esac
    [ "$a1" = "$CHAIN" ] && { echo "${p#/proc/}"; return 0; }
  done
  return 1
}
mtime(){ [ -f "$1" ] && stat -c %Y "$1" || echo 0; }

log "=========================================================="
log "감시 시작 (주기 ${INTERVAL}s, 정체판정 ${STALL}s, pgrep 미사용)"

last_m=0; last_chg=$(date +%s); restarts=0
while :; do
  if grep -q "$DONE_MARK" "$QLOG" 2>/dev/null; then
    nfail=$(grep -c "실패" "$QLOG" 2>/dev/null); nfail=${nfail:-0}
    log "✅ 큐 완료 (실패 단계 ${nfail}개)"
    [ "$nfail" -gt 0 ] && log "   ⚠️ 실패한 단계가 있으니 로그 확인 필요: grep 실패 $QLOG"
    exit 0
  fi

  pid=$(chain_pid || true)
  m=$(mtime "$QLOG")
  [ "$m" != "$last_m" ] && { last_m=$m; last_chg=$(date +%s); }
  age=$(( $(date +%s) - last_chg ))

  if [ -z "$pid" ]; then
    log "⚠️ 큐 프로세스 없음 (완료표시 없음) — 재기동 #$((++restarts))"
    TOPANEU_ROOT="$R" nohup bash "$CHAIN" >> "$QLOG" 2>&1 & disown
    last_chg=$(date +%s)
  elif [ "$age" -gt "$STALL" ]; then
    log "⚠️ 큐 정체 ${age}s (PID $pid, 로그 변화 없음) — 죽이고 재기동 #$((++restarts))"
    kill -9 "$pid" 2>/dev/null
    sleep 5
    TOPANEU_ROOT="$R" nohup bash "$CHAIN" >> "$QLOG" 2>&1 & disown
    last_chg=$(date +%s)
  fi

  if [ $(( $(date +%s) % 3600 )) -lt "$INTERVAL" ]; then
    log "(heartbeat) PID=${pid:-none} 로그정지 ${age}s 재기동 ${restarts}회 | $(tail -n1 "$QLOG" 2>/dev/null | cut -c1-80)"
  fi
  sleep "$INTERVAL"
done
