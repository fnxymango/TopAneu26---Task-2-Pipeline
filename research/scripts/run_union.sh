#!/bin/bash
# 합집합 게이트 e2e — (ICA&마진0.7) OR p2>0.25, X5 기반. 체인 DONE_ALL 후 실행.
# 규칙: X5+gC(test 0.3972/val 0.4725) 대비 test·val 둘 다 + 여야 채택.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
until grep -q "DONE_ALL" "$E/chain_status.md" 2>/dev/null; do sleep 30; done
echo "[$(TZ=Asia/Seoul date +%H:%M)] 합집합 게이트 시작" >> "$E/chain_status.md"
one(){ local sp=$1 sd=$2 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.25 TOPANEU_TOPK_OR=1 \
  CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "uni_${sp}_s${sd}" > "$E/uni_${sp}_s${sd}.log" 2>&1
}
for sd in 0 1 2 3 4; do one test $sd & one val $sd & done
wait
$PY - <<'PYEOF' >> "$E/chain_status.md" 2>&1
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
X5={"test":"c5_eval_test_x5g_test_s%d.json","val":"c5_eval_val_x5g_val_s%d.json"}
ok=True
print("\n### 합집합 게이트 (uni: gC OR p2>0.25)")
for sp in ("test","val"):
    fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_uni_{sp}_s*.json"))
    sds=[int(f.split("_s")[-1][:-5]) for f in fs]
    x=np.array([json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs])
    g=np.array([json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"]["MCC"] for s in sds])
    d=x-g; sd_=d.std(ddof=1) if len(d)>1 else 0
    t=d.mean()/(sd_/np.sqrt(len(d))) if sd_>1e-12 else float("nan")
    print(f"  {sp:5s} {x.mean():.4f}±{x.std(ddof=1):.4f}  vs x5g {g.mean():.4f}  Δ{d.mean():+.4f} t{t:+.2f} {int((d>0).sum())}/{len(d)}")
    if d.mean()<=0: ok=False
print(f"판정: {'PASS' if ok else 'FAIL'}")
open(A+"/chain_uni.verdict","w").write("PASS" if ok else "FAIL")
PYEOF
echo "UNION_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$E/chain_status.md"
