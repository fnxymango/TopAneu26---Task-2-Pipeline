#!/usr/bin/env bash
# b1watch — B1 체인 되살림 (crontab 5분). 죽은 단계만 다시 띄운다. 각 체인 최대 2회, 그 뒤엔 .gaveup_ 기록.
B=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/B1_vessel_contact_ft
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b1watch] $*" >> "$D/STATUS.log"; }
alive(){ "$PY" - "$1" <<'PYEOF'
import os, sys
want = sys.argv[1]; n = 0
for p in os.listdir('/proc'):
    if not p.isdigit():
        continue
    try:
        a = [x.decode(errors='ignore') for x in open(f'/proc/{p}/cmdline', 'rb').read().split(b'\0') if x]
    except Exception:
        continue
    if len(a) > 1 and os.path.basename(a[0]) == 'bash' and os.path.basename(a[1]) == want:
        n += 1
print(n)
PYEOF
}
retry(){  # $1 스크립트 · $2 마커
  local sh=$1 done=$2 c
  [ -f "$B/$done" ] && return 0
  [ "$(alive $sh)" != "0" ] && return 0
  [ -f "$B/.gaveup_$sh" ] && return 0
  c=$(cat "$B/.retry_$sh" 2>/dev/null || echo 0); c=$((c+1)); echo $c > "$B/.retry_$sh"
  if [ $c -gt 2 ]; then touch "$B/.gaveup_$sh"; log "$sh 2회 재시도 실패 → 포기"; return 0; fi
  log "$sh 죽어 있음 → $c 회차 되살림"
  cd "$B" && setsid nohup bash "$sh" >> "$B/${sh%.sh}.out" 2>&1 </dev/null &
}
retry B1T.sh .done_b1v
[ -f "$B/.done_b1v" ] && [ ! -f "$B/.skip_b1e" ] && retry B1E.sh .done_b1e
exit 0
