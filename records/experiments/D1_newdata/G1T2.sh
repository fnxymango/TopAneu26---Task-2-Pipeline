#!/usr/bin/env bash
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee; D=$R/experiments/D1_newdata; G=$R/experiments/G1_gc_neweval; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
exec 9>"$D/g1t2.lock"; flock -n 9 || exit 0; echo $$ > "$D/G1T2.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g1t] $*" | tee -a "$ST"; }
full(){ for sd in 0 1 2 4; do for t in b1gcON b1gcOFF; do [ "$(ls "$G/pred/${t}_test_s${sd}"/*.nii.gz 2>/dev/null | wc -l)" -ge 83 ] || return 1; done; done; }
log "test 시드 예측맵 8건 완성 대기"; for i in $(seq 1 240); do full && break; sleep 30; done
full || { log "★test 시드 예측맵 미완"; exit 1; }
log "test 시드 채점 8건 병렬 (split=test)"
for sd in 0 1 2 4; do for t in b1gcON b1gcOFF; do
  "$PY" "$D/neweval.py" "$G/pred/${t}_test_s${sd}" test "${t}_test_s${sd}" > "$G/scores_par/${t}_test_s${sd}.json" 2> "$G/logs/score_${t}_test_s${sd}.err" &
done; done; wait
log "G1T 완료"; touch "$D/.done_g1t"
