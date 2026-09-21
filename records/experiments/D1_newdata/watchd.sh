#!/usr/bin/env bash
# watchd — watch5.sh 를 5분마다 돌려 파일에 남기는 **세션 독립** 감시자.
#
# 왜 따로 두나: 5분 감시는 대화 세션에 묶여 있어 세션이 끊기면 같이 죽는다.
# 실험 자체는 setsid 로 살아남지만, 그 사이에 죽거나 멈춘 순간을 아무도 못 본다.
# 이 스크립트는 setsid 로 띄워 세션과 무관하게 돌며, 재접속 후 WATCH.log 만 보면
# 끊긴 구간에 무슨 일이 있었는지 복원된다.
#
# 종료 조건: H2·H4 가 둘 다 완료표시를 남기면 스스로 끝난다(무한정 돌지 않는다).
set -uo pipefail
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
LOG=$D/WATCH.log
exec 8>"$D/watchd.lock"; flock -n 8 || { echo "이미 실행 중"; exit 0; }
for i in $(seq 1 288); do            # 5분 × 288 = 24시간 상한
  { bash "$D/watch5.sh"; } >> "$LOG" 2>&1
  # 이상 징후(★)가 있으면 눈에 띄게 따로 모아둔다
  tail -20 "$LOG" | grep -q '★' && tail -20 "$LOG" | grep '★' | sed "s/^/[$(TZ=Asia/Seoul date +%H:%M)] /" >> "$D/WATCH_ALERT.log"
  if [ -f "$D/.done_h2" ] && [ -f "$D/.done_h4" ]; then
    echo "== $(TZ=Asia/Seoul date +%H:%M:%S) KST · H2·H4 모두 완료 — 감시 종료 ==" >> "$LOG"; break
  fi
  sleep 300
done
