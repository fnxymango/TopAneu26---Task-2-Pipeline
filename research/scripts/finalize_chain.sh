#!/bin/bash
# F1 최종화 (2026-08-26): UNION_DONE 후 — 승자 선택 -> 시드 5~9 추가(10시드) -> 6지표+부트스트랩 기록.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
ST=$E/chain_status.md
until grep -q "UNION_DONE" "$ST" 2>/dev/null; do sleep 30; done
V=$(cat "$A/chain_uni.verdict" 2>/dev/null || echo FAIL)
if [ "$V" = PASS ]; then WTAG=uni; WENV="TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.25 TOPANEU_TOPK_OR=1"
else WTAG=x5g; WENV="TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7"; fi
echo "[$(TZ=Asia/Seoul date +%H:%M)] F1 최종화 시작 — 승자=$WTAG" >> "$ST"
one(){ local sp=$1 sd=$2 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  [ -f "$A/c5_eval_${sp}_${WTAG}_${sp}_s${sd}.json" ] && return 0
  env $WENV CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${WTAG}_${sp}_s${sd}" > "$E/${WTAG}_${sp}_s${sd}.log" 2>&1
}
for sd in 5 6 7 8 9; do one test $sd & one val $sd & done
wait
$PY - "$WTAG" <<'PYEOF' >> "$ST" 2>&1
import json,glob,math,sys,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
W=sys.argv[1]
print(f"\n### F1 최종 수치 — {W} (10시드)")
for sp in ("test","val"):
    fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_{W}_{sp}_s*.json"))
    for blk,lab in (("adjusted_div_present","cov"),("official_div52","off52")):
        r={k:np.array([json.load(open(f))[blk][k] for f in fs]) for k in ("MCC","PRECISION","RECALL","DICE","HD95","VOLSIM")}
        print(f"  [{sp} {lab}] n={len(fs)} "+"  ".join(f"{k} {v.mean():.4f}±{v.std(ddof=1):.4f}" for k,v in r.items()))
# 케이스 부트스트랩 (존재클래스 재계산판)
KEYS=["TP","FP","FN","TN"]
def cov(pc,idx,eps=1e-6):
    S={k:np.zeros(53) for k in KEYS}
    for i in idx:
        c=pc[i]
        for k in KEYS:
            for j in range(1,53): S[k][j]+=c[f"{k}_{j}"]
    out=[]
    for j in range(1,53):
        tp,fp,fn,tn=S["TP"][j],S["FP"][j],S["FN"][j],S["TN"][j]
        if tp+fn==0: continue
        out.append((tp*tn-fn*fp)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+eps))
    return float(np.mean(out)) if out else 0.0
rng=np.random.default_rng(0); absv=[]
for f in sorted(glob.glob(f"{A}/c5_percase_test_{W}_test_s*.json"))[:5]:
    pc=json.load(open(f))["per_case"]; n=len(pc)
    for _ in range(200): absv.append(cov(pc,rng.integers(0,n,n)))
a=np.array(absv)
print(f"  test 케이스 부트스트랩(존재클래스 재계산): 중앙 {np.median(a):.4f}  90% CI [{np.percentile(a,5):.4f}, {np.percentile(a,95):.4f}]")
PYEOF
echo "FINALIZE_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$ST"
