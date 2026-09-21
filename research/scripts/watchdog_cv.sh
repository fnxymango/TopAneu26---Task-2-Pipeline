#!/usr/bin/env bash
# CV 감시견 — 3분마다 상태 점검, 죽으면 되살림 (세션 독립).
#   · 드라이버가 살아 있으면 진행만 기록
#   · 드라이버가 죽었고 결과가 없으면 → 동시 실행 수를 낮춰 v2 드라이버 재기동
#   · RAM 여유가 부족하면 다음 재기동 시 동시 수를 더 낮춤
#   · cv_pp.json이 생기면 종료
set -uo pipefail
BASE=/home/user/TopAneu/seg/sblee/nnunet
N=$BASE/experiments/D600_vessel_skelrec_resencm_250ep_bd0
LOG=$N/watchdog.log
CV=$N/cv
INTERVAL=180
MAXP=${START_MAXP:-8}
RESTARTS=0
MAX_RESTARTS=6

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
drv() { pgrep -u "$USER" -f "run_cv_5fold" | grep -v "^$$\$" | wc -l; }
done_n() { local t=0 f h; for f in 1 2 3 4; do h=$(ls "$CV/fold$f/out"/*.nii.gz 2>/dev/null | wc -l); t=$((t+h)); done; echo $t; }

: > "$LOG"
say "감시 시작 — ${INTERVAL}초 주기, 최대 재기동 ${MAX_RESTARTS}회"

while true; do
  if [ -f "$N/cv_raw.json" ] && [ -f "$N/cv_pp.json" ]; then
    say "[완료] CV 결과 파일 생성 확인 — 감시 종료"; exit 0
  fi

  NPRED=$(pgrep -u "$USER" -fc predict_seq.py 2>/dev/null || echo 0)
  NDRV=$(drv)
  DONE=$(done_n)
  AVAIL=$(free -g | awk '/^Mem:/{print $7}')
  say "예측 ${DONE}/69 · predict_seq ${NPRED}개 · 드라이버 ${NDRV} · RAM여유 ${AVAIL}GB"

  # RAM이 위험하면 다음 재기동 동시 수를 낮춤
  if [ "$AVAIL" -lt 8 ] && [ "$MAXP" -gt 4 ]; then
    MAXP=$((MAXP-2)); say "[!] RAM 여유 ${AVAIL}GB — 다음 재기동 동시 수 ${MAXP}로 하향"
  fi

  if [ "$NDRV" -eq 0 ]; then
    if [ "$DONE" -ge 69 ]; then
      say "[!] 드라이버 없음 · 예측은 완료 — 채점 단계 재기동 (동시 $MAXP)"
    else
      say "[!] 드라이버 없음 · 예측 ${DONE}/69 — 재기동 (동시 $MAXP)"
    fi
    RESTARTS=$((RESTARTS+1))
    if [ "$RESTARTS" -gt "$MAX_RESTARTS" ]; then
      say "[!] 재기동 ${MAX_RESTARTS}회 초과 — 감시 종료(수동 확인 필요)"; exit 1
    fi
    CV_MAXPROC=$MAXP setsid nohup "$BASE/scripts/run_cv_5fold_v2.sh" \
      </dev/null >> "$BASE/experiments/nohup_cv_5fold.out" 2>&1 &
    sleep 20
  fi
  sleep "$INTERVAL"
done
