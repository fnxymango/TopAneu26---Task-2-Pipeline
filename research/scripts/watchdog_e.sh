#!/usr/bin/env bash
# E계열 감시자 (2026-08-19). E1 이 혈관 환경 문제로 조용히 죽었는데 기존 watchdog 이
# A8/C44 만 알아서 아무도 못 잡았다. 그래서 E계열 전용 감시자를 따로 둔다.
#
# 프로세스 탐지는 /proc 의 **argv[1] 완전일치**로만 한다. cmdline 전체를 패턴매칭하면
# 감시자 자신의 bash 래퍼가 걸려 자기를 죽인다(2026-08-17 실제 사고).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"
PERIOD=180
log(){ echo "[wqe $(date -u +'%m-%d %H:%M:%S')] $*"; }

alive(){ # $1=스크립트 파일명
  for p in /proc/[0-9]*; do
    a1=$(tr '\0' '\n' < "$p/cmdline" 2>/dev/null | sed -n 2p) || continue
    [ "$(basename "$a1" 2>/dev/null)" = "$1" ] && return 0
  done
  return 1
}
done_e1(){  [ "$(ls "$E/_c4_bpgraph/vespp_train"/*.json 2>/dev/null | wc -l)" -ge 285 ]; }
done_e4(){  [ "$(ls "$A"/c5_eval_test_e4_*.json 2>/dev/null | wc -l)" -ge 10 ]; }
done_e4b(){ [ "$(ls "$A"/c5_eval_test_e4b_*.json 2>/dev/null | wc -l)" -ge 20 ]; }
done_e1b(){ [ "$(ls "$A"/c5_eval_train_e1b_*.json 2>/dev/null | wc -l)" -ge 15 ]; }
done_e6(){  [ -f "$A/e6_bootstrap_test.json" ]; }

log "=========================================================="
log "E계열 감시 시작 (주기 ${PERIOD}s, argv[1] 완전일치, pgrep 미사용)"
while :; do
  for pair in "chain_e1.sh:done_e1" "chain_e4.sh:done_e4" "chain_e4b.sh:done_e4b" \
              "chain_e6.sh:done_e6" "chain_e6b.sh:done_e6" "chain_e1b.sh:done_e1b" \
              "chain_e7.sh:false"; do
    sc="${pair%%:*}"; chk="${pair##*:}"
    alive "$sc" && continue
    [ "$chk" != "false" ] && $chk && continue          # 할 일을 마치고 끝난 것
    log "⚠ $sc 가 죽었는데 산출물이 미완 — 재시작"
    ( cd "$S" && setsid nohup env TOPANEU_ROOT="$R" bash "./$sc" \
        >> "$E/$(basename "$sc" .sh)_chain.log" 2>&1 < /dev/null & disown ) || log "  재시작 실패"
    sleep 10
  done
  if done_e1 && done_e4 && done_e4b && done_e1b && done_e6; then
    log "✅ E계열 전 단계 완료 — 감시 종료"; break
  fi
  sleep "$PERIOD"
done
