#!/bin/bash
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
one(){ local sp=$1 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=3 RF_TREES=500 OMP_NUM_THREADS=4 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-model "$A/final_rf_seed3.pkl" --save-pred-dir "$E/final_pred_seed3_$sp" \
    --tag "freeze3_${sp}" > "$E/freeze3_${sp}.log" 2>&1
}
one test & one val & wait
$PY - <<'PYEOF'
import json
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
for sp,ref in (("test","c5_eval_test_x5g_test_s3.json"),("val","c5_eval_val_x5g_val_s3.json")):
    a=json.load(open(f"{A}/c5_eval_{sp}_freeze3_{sp}.json"))["adjusted_div_present"]; b=json.load(open(f"{A}/{ref}"))["adjusted_div_present"]
    print(f"[freeze3 {sp}] cov.MCC {a['MCC']:.6f} vs 기록 시드3 {b['MCC']:.6f}  오차 {a['MCC']-b['MCC']:+.2e}")
PYEOF
echo "FREEZE3_DONE $(TZ=Asia/Seoul date +%H:%M) KST"
