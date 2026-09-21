#!/bin/bash
# vote3f 검출본 e2e — test / val 각 5시드. 규칙: test Δ>0 AND val Δ>0 이어야 채택.
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$R/experiments/_c1_realpred; BP=$R/experiments/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S

run() {  # split vesdir bpdir aneudir tag
  for SD in 0 1 2 3 4; do
    CLF_SEED=$SD OMP_NUM_THREADS=3 $PY -u c5_location_v2.py eval \
      --train-feat "$A/e11_feat_hyb_ov.json" --split "$1" \
      --vessel-dir "$P/$2" --bp-dir "$BP/$3" --aneurysm-pred-dir "$P/$4" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "$5_s${SD}" > "$R/experiments/$5_s${SD}.log" 2>&1
  done
}
run test vespp_test vespp_test aneu_test_vote3f v3f_test & T=$!
run val  vespp_val  val_pred   aneu_val_vote3f  v3f_val  & V=$!
wait $T $V
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
