#!/bin/bash
# Q7 다중라벨 방출 — 2등/3등 클래스에 복셀 조각만 떼어 준다.
# 채택 규칙(결과 보기 전 고정): test Δcov.MCC > 0 AND val Δcov.MCC > 0.
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$R/experiments/_c1_realpred; BP=$R/experiments/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S

one() {   # split K VOX TAU seed tag
  local sp=$1 k=$2 vx=$3 tau=$4 sd=$5 tg=$6
  local ves bp an
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; an=aneu_test_probavgf
  else ves=vespp_val; bp=val_pred; an=aneu_val_probavgf; fi
  TOPANEU_TOPK=$k TOPANEU_TOPK_VOX=$vx TOPANEU_TOPK_TAU=$tau CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval \
    --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$an" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tg}_s${sd}" > "$R/experiments/${tg}_s${sd}.log" 2>&1
}

# 0) 회귀검증: K=1 이면 기존 regress_off(seed0 cov.MCC 0.3860) 와 같아야 한다
one test 1 3 0 0 k1_regress

lane() { for sd in 0 1 2; do one "$1" "$2" "$3" "$4" $sd "$5"; done; }
#     split K VOX TAU tag
lane test 2 3 0   tk2_test  & L1=$!
lane test 3 3 0   tk3_test  & L2=$!
lane test 2 3 0.5 tk2t_test & L3=$!
lane val  2 3 0   tk2_val   & L4=$!
lane val  3 3 0   tk3_val   & L5=$!
lane val  2 3 0.5 tk2t_val  & L6=$!
wait $L1 $L2 $L3 $L4 $L5 $L6
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
