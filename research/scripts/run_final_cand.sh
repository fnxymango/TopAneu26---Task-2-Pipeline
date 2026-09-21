#!/bin/bash
# 최종 설정 후보: gC (ICA·마진0.7 2등조각) + RF_TREES=1500.
# 채택 규칙(고정): gC 단독 대비 test·val 둘 다 나빠지지 않아야 트리업만 최종에 포함.
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$R/experiments/_c1_realpred; BP=$R/experiments/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
one() {  # split seed
  local sp=$1 sd=$2 ves bp an
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; an=aneu_test_probavgf
  else ves=vespp_val; bp=val_pred; an=aneu_val_probavgf; fi
  RF_TREES=1500 TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$an" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "fin_${sp}_s${sd}" > "$R/experiments/fin_${sp}_s${sd}.log" 2>&1
}
# 병렬: 시드×스플릿 전부 동시 (10 프로세스, 코어 여유 있음)
for sd in 0 1 2 3 4; do one test $sd & one val $sd & done
wait
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
