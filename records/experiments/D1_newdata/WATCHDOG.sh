#!/usr/bin/env bash
# 감시견 v3 — 단계 체인 B1 → B2 → B2EVAL → ENS 를 완주시킨다.
#   · 각 단계 스크립트는 선행 단계의 .done 마커를 스스로 기다리므로, 전부 살아만 있으면 된다
#   · 죽으면: 전제조건(preflight) 통과 → 직전과 다른 실패 서명 → 3회 이내 일 때만 재시동
#   · 그 외엔 DIAGNOSIS_<단계>.md 를 쓰고 .halted 로 멈춘다 (무한 재시도 금지)
#   · 감시견 자신은 crontab(5분) 이 되살린다 — resurrect.sh
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; ST=$D/STATUS.log; WL=$D/watchdog.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
MAXR=3; INT=120
STAGES="B1 B2 B2EVAL ENS"
declare -A DONE=( [B1]=.done_b1 [B2]=.done_b2 [B2EVAL]=.done_b2eval [ENS]=.done_ens )
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][watch] $*" | tee -a "$WL" >> "$ST"; }
exec 8>"$D/watchdog.lock"; flock -n 8 || exit 0
echo $$ > "$D/WATCHDOG.pid"

alive(){
  local pf="$D/$1.pid" pid cl
  [ -f "$pf" ] || return 1
  pid=$(cat "$pf" 2>/dev/null) || return 1
  case "$pid" in ''|*[!0-9]*) return 1;; esac
  [ -r "/proc/$pid/cmdline" ] || return 1
  cl=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
  grep -xF "$D/$1.sh" <<< "$cl" > /dev/null
}
sig_of(){
  { grep '★' "$ST" 2>/dev/null | tail -1
    for f in "$D"/*.out* "$R"/experiments/B*/logs/*.log; do
      [ -f "$f" ] && tail -40 "$f" 2>/dev/null | grep -aE "^[A-Za-z_.]*(Error|Exception):" | tail -1
    done
  } 2>/dev/null | md5sum | cut -c1-12
}
diagnose(){
  local n=$1 why=$2
  {
    echo "# 자동 진단 — $(TZ=Asia/Seoul date +'%Y-%m-%d %H:%M') KST"; echo
    echo "## 중단: \`$n\` · $why"; echo
    echo "## 전제조건"; echo '```'; cat "$D/preflight_last.txt" 2>/dev/null; echo '```'; echo
    echo "## 최근 실패 흔적"; echo '```'; grep '★' "$ST" | tail -5; echo '```'; echo
    echo "## 로그 꼬리"
    for f in "$D/$n.out"*; do [ -f "$f" ] && { echo "### $(basename "$f")"; echo '```'; tail -25 "$f" | tr -d '\r'; echo '```'; }; done
    echo; echo "## 복구"; echo "원인 해결 후:  rm -f $D/.halted_$n $D/.sig_$n   → 감시견이 다시 띄운다"
  } > "$D/DIAGNOSIS_$n.md"
  touch "$D/.halted_$n"
  log "★$n 중단 · $why · $D/DIAGNOSIS_$n.md"
}
declare -A N
restart(){
  local n=$1 s prev
  N[$n]=${N[$n]:-0}
  [ -f "$D/.halted_$n" ] && return 1
  if ! "$PY" "$D/preflight.py" > "$D/preflight_last.txt" 2>&1; then
    diagnose "$n" "전제조건 미충족"; return 1; fi
  s=$(sig_of); prev=$(cat "$D/.sig_$n" 2>/dev/null || true)
  if [ -n "$prev" ] && [ "$s" = "$prev" ]; then
    diagnose "$n" "직전과 동일한 실패 재현 (서명 $s)"; return 1; fi
  echo "$s" > "$D/.sig_$n"
  if [ "${N[$n]}" -ge "$MAXR" ]; then diagnose "$n" "재시동 ${MAXR}회 초과"; return 1; fi
  N[$n]=$(( N[$n] + 1 ))
  rm -f "$D/$(echo "$n" | tr 'A-Z' 'a-z').lock"
  log "$n 죽음 → 전제조건 통과, 재시동 (${N[$n]}/$MAXR)"
  setsid nohup "$D/$n.sh" > "$D/$n.out.$(date +%s)" 2>&1 </dev/null & disown
  sleep 15
}
log "감시견 v3 기동 (PID $$) · 단계 $STAGES · ${INT}초 주기"
while :; do
  alldone=1
  for n in $STAGES; do
    [ -f "$D/${DONE[$n]}" ] && continue
    alldone=0
    [ -f "$D/.halted_$n" ] && continue
    alive "$n" || restart "$n"
  done
  [ "$alldone" = 1 ] && { log "전 단계 완료 — 감시 종료"; touch "$D/.done_all"; break; }
  sleep "$INT"
done
