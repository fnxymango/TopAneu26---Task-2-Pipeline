#!/usr/bin/env bash
# crontab 이 5분마다 호출. 감시견이 죽어 있으면 되살린다. 전 단계 완료면 아무것도 안 한다.
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
# G2(gC on/off · CPU 후처리) 되살림
if [ ! -f "$D/.done_g2" ]; then
  gp=$(cat "$D/G2.pid" 2>/dev/null)
  if ! { [ -n "$gp" ] && [ -r "/proc/$gp/cmdline" ] && tr '\0' '\n' < "/proc/$gp/cmdline" | grep -qxF "$D/G2.sh"; }; then
    rm -f "$D/g2.lock"; echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][cron] G2 없음 → 되살림" >> "$D/STATUS.log"
    setsid nohup "$D/G2.sh" > "$D/G2.out.cron.$(date +%s)" 2>&1 </dev/null & disown
  fi
fi
[ -f "$D/.done_all" ] && exit 0
pid=$(cat "$D/WATCHDOG.pid" 2>/dev/null)
if [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] && tr '\0' '\n' < "/proc/$pid/cmdline" | grep -qxF "$D/WATCHDOG.sh"; then exit 0; fi
rm -f "$D/watchdog.lock"
echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][cron] 감시견 없음 → 되살림" >> "$D/STATUS.log"
setsid nohup "$D/WATCHDOG.sh" > "$D/WATCHDOG.out.cron.$(date +%s)" 2>&1 </dev/null & disown
