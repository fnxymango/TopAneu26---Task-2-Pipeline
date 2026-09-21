#!/usr/bin/env bash
# H2FILL — H2.sh 의 배치 wait 때문에 비는 코어를 채운다.
# H2.sh 는 14개씩 묶어 wait 하므로 먼저 끝난 슬롯이 다음 배치까지 논다.
# H2.sh 가 **가장 나중에** 도달할 항목부터 역순으로 미리 채점해 두면, 도달했을 때 건너뛴다.
# 임시파일 → mv 로 원자적 배치. 중복 실행이 겹쳐도 최종 파일은 유효한 한 벌이 된다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; H=$R/experiments/H1_patchfilter
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
exec 9>"$D/h2fill.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
one(){ local tag=$1 sp=$2 sd=$3 t
  [ -s "$H/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$H/scores/${tag}_${sp}_s${sd}.json" && return 0
  t="$H/scores/.fill_${tag}_${sp}_s${sd}.$$"
  "$PY" "$D/neweval.py" "$H/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" \
    > "$t" 2> "$H/logs/fill_${tag}_${sp}_s${sd}.err"
  if grep -q '"label"' "$t"; then mv -f "$t" "$H/scores/${tag}_${sp}_s${sd}.json"; echo "  완료 ${tag}_${sp}_s${sd}"
  else rm -f "$t"; echo "  ★실패 ${tag}_${sp}_s${sd}"; fi
}
n=0
for sd in 4 3 2 1 0; do for spec in "b1Noff test" "b1Noff val"; do set -- $spec
  one $1 $2 $sd & n=$((n+1)); [ $((n%6)) -eq 0 ] && wait
done; done
wait
echo "H2FILL 완료"
