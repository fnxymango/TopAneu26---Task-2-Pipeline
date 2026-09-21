#!/usr/bin/env bash
# K1·K2 중 포기된 체인이 있으면 Q5 대기 조건(.done_k1 · .done_k2)을 풀어주는 연결 마커를 만든다(원인 기록 포함).
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
for t in k1 k2; do
  if [ -f "$D/.gaveup_$t" ] && [ ! -f "$D/.done_$t" ]; then
    echo "포기로 대체된 완료 마커 — $(TZ=Asia/Seoul date +'%m-%d %H:%M') KST · 결과 없음" > "$D/.done_$t"
    echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][q5watch] $t 포기 → Q5 대기 해제용 .done_$t(내용에 '포기' 표기)" >> "$D/STATUS.log"
  fi
done
