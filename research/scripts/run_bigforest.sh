#!/bin/bash
# 제출 후보 기록: RF 5000그루 (10시드 앙상블 등가) × X5+gC — 시드 복권 제거판. 시드 0·1 두 개로 재현성만 확인.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
one(){ local sp=$1 sd=$2 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  RF_TREES=5000 TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=5 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "big5k_${sp}_s${sd}" > "$E/big5k_${sp}_s${sd}.log" 2>&1
}
for sd in 0 1; do one test $sd & one val $sd & done
wait
$PY - <<'PYEOF' >> "$E/chain_status.md" 2>&1
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
print("\n### 제출후보 기록 — RF 5000그루 × X5+gC (시드복권 제거판)")
for sp in ("test","val"):
    fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_big5k_{sp}_s*.json"))
    for f in fs:
        d=json.load(open(f)); c=d["adjusted_div_present"]; o=d["official_div52"]
        print(f"  {sp:5s} {f.split('_s')[-1][:-5]:>2}  cov.MCC {c['MCC']:.4f}  off52 {o['MCC']:.4f}  (P {c['PRECISION']:.3f} R {c['RECALL']:.3f})")
    v=[json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs]
    if len(v)>1: print(f"  {sp:5s} 시드간 차 {abs(v[0]-v[1]):.4f}  (500그루 10시드 sd 는 test 0.019 / val 0.023)")
PYEOF
echo "BIG5K_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$E/chain_status.md"
