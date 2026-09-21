#!/usr/bin/env bash
# promote_fill — 고아가 된 .fill_* 임시 채점결과를 정식 파일로 승격한다.
# H2FILL 의 백그라운드 서브셸을 죽이면서 mv 단계가 사라졌다. neweval 자체는 계속 돌아
# 임시파일을 완성하므로, 유효해지는 대로(=키 "label" 이 들어오는 대로) 옮겨준다.
set -uo pipefail
H=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/H1_patchfilter
for i in $(seq 1 120); do
  left=0
  for f in "$H"/scores/.fill_*; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); b=${b#.fill_}; tgt="$H/scores/${b%.*}.json"
    if grep -q '"label"' "$f" 2>/dev/null; then
      if [ -s "$tgt" ] && grep -q '"label"' "$tgt" 2>/dev/null; then rm -f "$f"; echo "  중복 폐기 $(basename "$f")"
      else mv -f "$f" "$tgt"; echo "  승격 $(basename "$tgt")"; fi
    else left=$((left+1)); fi
  done
  [ "$left" -eq 0 ] && { echo "남은 임시 0 — 종료"; break; }
  sleep 30
done
