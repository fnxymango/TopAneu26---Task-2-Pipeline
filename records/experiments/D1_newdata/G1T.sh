#!/usr/bin/env bash
# G1T — 본판정(test, 번들 검출기) 의 시드 강건성: seed 0,1,2,4 로 gC on/off 를 새 eval 채점 (seed3 은 이미 있음)
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; G=$E/G1_gc_neweval; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/g1t.lock"; flock -n 9 || exit 0; echo $$ > "$D/G1T.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g1t] $*" | tee -a "$ST"; }
cd "$S"; mkdir -p "$G"/{pred,logs,scores_par}
gen(){ local sd=$1 topk=$2 tag=$3
  TOPANEU_TOPK=$topk TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split test --fast \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" --aneurysm-pred-dir "$P/aneu_test_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$G/pred/${tag}_test_s${sd}" --tag "g1s_${tag}_test_s${sd}" > "$G/logs/${tag}_test_s${sd}.log" 2>&1
}
log "시드 강건성: seed 0,1,2,4 × gC on/off (test, 번들 검출기) 생성"
for sd in 0 1 2 4; do gen $sd 2 b1gcON & gen $sd 1 b1gcOFF & done; wait
log "채점 8건 병렬"
for sd in 0 1 2 4; do for t in b1gcON b1gcOFF; do
  "$PY" "$D/neweval.py" "$G/pred/${t}_test_s${sd}" test "${t}_test_s${sd}" > "$G/scores_par/${t}_test_s${sd}.json" 2> "$G/logs/score_${t}_test_s${sd}.err" &
done; done; wait
log "G1T 완료"; touch "$D/.done_g1t"
