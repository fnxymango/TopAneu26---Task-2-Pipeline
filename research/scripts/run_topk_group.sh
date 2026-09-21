#!/bin/bash
# Q7b 클래스군 게이트 — "잘 맞히는 클래스는 절대 안 건드린다".
# 채택 규칙(고정): test Δcov.MCC > 0 AND val Δcov.MCC > 0.
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$R/experiments/_c1_realpred; BP=$R/experiments/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
one() {  # split ICA MAXN MRG seed tag
  local sp=$1 ic=$2 mx=$3 mg=$4 sd=$5 tg=$6 ves bp an
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; an=aneu_test_probavgf
  else ves=vespp_val; bp=val_pred; an=aneu_val_probavgf; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=$ic TOPANEU_TOPK_MAXN=$mx TOPANEU_TOPK_MARGIN=$mg \
  CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$an" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tg}_s${sd}" > "$R/experiments/${tg}_s${sd}.log" 2>&1
}
lane() { for sd in 0 1 2 3 4; do one "$1" "$2" "$3" "$4" $sd "$5"; done; }
#    split ICA MAXN MRG  tag
lane test 0 4  0    gA_test & lane val 0 4  0    gA_val &
lane test 1 0  0.5  gB_test & lane val 1 0  0.5  gB_val &
lane test 1 0  0.7  gC_test & lane val 1 0  0.7  gC_val &
wait
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
