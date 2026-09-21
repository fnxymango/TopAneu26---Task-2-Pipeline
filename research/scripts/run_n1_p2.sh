#!/bin/bash
# N1 절대확률 게이트 e2e — X5 기반, p2>0.20 / 0.25. 규칙: X5+gC(test 0.3972/val 0.4725) 를
# test·val 둘 다 넘어야 gC 를 대체한다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
until grep -q "^DONE" "$E/q15_surx5.log" 2>/dev/null; do sleep 30; done
echo "[STEP] $(TZ=Asia/Seoul date +%H:%M) q15 종료 — N1 시작"
one() {  # split P2 seed tag
  local sp=$1 p2=$2 sd=$3 tg=$4 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_P2=$p2 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tg}_s${sd}" > "$E/${tg}_s${sd}.log" 2>&1
}
for sd in 0 1 2 3 4; do
  one test 0.20 $sd n1a_test & one val 0.20 $sd n1a_val &
  one test 0.25 $sd n1b_test & one val 0.25 $sd n1b_val &
done
wait
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
