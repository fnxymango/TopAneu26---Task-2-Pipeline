#!/bin/bash
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$R/experiments/_c1_realpred; BP=$R/experiments/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
one() {  # split K MRG seed tag
  local sp=$1 k=$2 mg=$3 sd=$4 tg=$5 ves bp an
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; an=aneu_test_probavgf
  else ves=vespp_val; bp=val_pred; an=aneu_val_probavgf; fi
  TOPANEU_TOPK=$k TOPANEU_TOPK_MARGIN=$mg CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$an" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tg}_s${sd}" > "$R/experiments/${tg}_s${sd}.log" 2>&1
}
lane() { for sd in 0 1 2 3 4; do one "$1" "$2" "$3" $sd "$4"; done; }
lane test 2 0.70 mg70_test & lane test 2 0.90 mg90_test &
lane val  2 0.70 mg70_val  & lane val  2 0.90 mg90_val  &
wait
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
