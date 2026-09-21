#!/usr/bin/env bash
# resume_neweval — 일시정지시킨 H5 채점을 반드시 되살린다.
# 조건: H4 가 3단계(c5)를 벗어나면(=4단계 로그가 뜨면) 즉시. 아니면 60분 뒤 무조건.
# 멈춘 채 잊히는 것이 가장 위험하므로 타임아웃을 둔다.
set -uo pipefail
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PIDS="1400 1402 1405 1407 1408 "
for i in $(seq 1 180); do
  if grep -q '\[h4\] 4단계' "$D/STATUS.log" 2>/dev/null || [ -f "$D/.done_h4" ]; then
    echo "[$(TZ=Asia/Seoul date +%H:%M)] H4 3단계 종료 감지 — 재개"; break
  fi
  sleep 20
done
for p in $PIDS; do kill -CONT "$p" 2>/dev/null && echo "  재개 $p"; done
echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h5] 일시정지했던 채점 5런 재개" >> "$D/STATUS.log"
