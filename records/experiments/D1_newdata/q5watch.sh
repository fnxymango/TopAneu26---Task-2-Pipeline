#!/usr/bin/env bash
# q5watch — crontab 5분마다. 2026-09-15 큐(K1 → K2 → Q5[→T3] · K1R · K3P) 감시 · 세션 독립.
#  · 체인이 끝나지 않았는데 프로세스가 없으면 되살린다(각 스크립트는 끝난 런을 건너뛰므로 재실행 안전). 체인당 최대 2회.
#  · 2회 되살려도 죽으면 '포기' 마커(.gaveup_<체인>)를 남긴다 → Q5 는 .done 또는 .gaveup 둘 중 하나면 진행.
#  · 프로세스 확인은 /proc argv 정확 대조(bash <스크립트명>). 전부 끝나면 아무것도 안 한다.
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
cd "$D" || exit 1
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][q5watch] $*" >> "$D/STATUS.log"; }
alive(){ for p in /proc/[0-9]*; do
  a0=$(tr '\0' '\n' < $p/cmdline 2>/dev/null | sed -n 1p); a1=$(tr '\0' '\n' < $p/cmdline 2>/dev/null | sed -n 2p)
  [ "$a0" = bash ] && [ "$a1" = "$1" ] && return 0; done 2>/dev/null; return 1; }
for spec in "K1.sh k1" "K2.sh k2" "Q5.sh q5" "K1R.sh k1r" "K3P.sh k3p"; do set -- $spec; sh=$1; t=$2
  [ -f ".done_$t" ] || [ -f ".gaveup_$t" ] && continue
  alive "$sh" && continue
  # T3.sh 는 Q5 가 직접 부른다 — Q5 가 살아 있으면 T3 도 그 아래에서 돈다
  n=$(cat ".retry_$t" 2>/dev/null || echo 0)
  if [ "$n" -ge 2 ]; then touch ".gaveup_$t"; log "$sh 2회 되살려도 종료 → 포기 마커(.gaveup_$t) · 뒤 큐는 진행"; continue; fi
  echo $((n+1)) > ".retry_$t"; rm -f "$t.lock"
  log "$sh 프로세스 없음(완료 마커 없음) → 되살림 $((n+1))/2"
  setsid nohup bash "$sh" >> "${sh%.sh}.out" 2>&1 < /dev/null & disown
done
