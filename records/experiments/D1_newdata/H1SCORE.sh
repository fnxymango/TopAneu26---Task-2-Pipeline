#!/usr/bin/env bash
# H1SCORE — 3단계 채점을 뒤쪽 팔부터 역순으로 추가 병렬화한다.
# H1.sh 의 score() 는 파일이 이미 있고 '"label"' 을 담고 있으면 건너뛰므로,
# 여기서 먼저 완성해두면 H1.sh 가 그 팔에 도달했을 때 그냥 넘어간다.
# 임시파일에 쓰고 mv 로 원자적 배치 — H1.sh 가 반쯤 쓰인 파일을 유효하다고 볼 여지를 없앤다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; H=$R/experiments/H1_patchfilter
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
exec 9>"$D/h1score.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
one(){ local tag=$1 sp=$2 sd=$3 t
  t="$H/scores/.tmp_${tag}_${sp}_s${sd}.$$"
  [ -s "$H/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$H/scores/${tag}_${sp}_s${sd}.json" && return 0
  "$PY" "$D/neweval.py" "$H/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" \
    > "$t" 2> "$H/logs/score2_${tag}_${sp}_s${sd}.err"
  if grep -q '"label"' "$t"; then mv -f "$t" "$H/scores/${tag}_${sp}_s${sd}.json"
  else rm -f "$t"; echo "★실패 ${tag}_${sp}_s${sd}"; fi
}
n=0
for t in b1on_pf b1on e9off_pf; do for sp in val test; do for sd in 4 3 2 1 0; do
  one $t $sp $sd & n=$((n+1)); [ $((n%8)) -eq 0 ] && wait
done; done; done; wait
echo "H1SCORE 완료"
