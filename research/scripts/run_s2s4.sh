#!/bin/bash
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
one(){ local sp=$1 sd=$2 tg=$3 feat=$4; shift 4; local ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  env "$@" TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/$feat" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tg}_${sp}_s${sd}" > "$E/${tg}_${sp}_s${sd}.log" 2>&1
}
for sd in 0 1 2 3 4; do
  one test $sd s4d1 e11_feat_hyb_ov.json TOPANEU_OUT_DILATE=1 &
  one val  $sd s4d1 e11_feat_hyb_ov.json TOPANEU_OUT_DILATE=1 &
  one test $sd s2cl e11_feat_hyb_ov_clean.json &
  one val  $sd s2cl e11_feat_hyb_ov_clean.json &
done
wait
$PY - <<'PYEOF' >> "$E/chain_status.md" 2>&1
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
X5={"test":"c5_eval_test_x5g_test_s%d.json","val":"c5_eval_val_x5g_val_s%d.json"}
K=["MCC","PRECISION","RECALL","DICE","HD95","VOLSIM"]
for tag,lab in (("s4d1","S4 출력팽창 +1복셀"),("s2cl","S2 의심라벨 제거")):
    print(f"\n### {lab} ({tag}) vs X5+gC")
    ok=True
    for sp in ("test","val"):
        fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_{tag}_{sp}_s*.json")); sds=[int(f.split("_s")[-1][:-5]) for f in fs]
        if not fs: print("  결과 없음"); ok=False; continue
        line=[]
        for k in K:
            x=np.array([json.load(open(f))["adjusted_div_present"][k] for f in fs])
            g=np.array([json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"][k] for s in sds])
            line.append(f"{k} {x.mean():.4f}({x.mean()-g.mean():+.4f})")
        dm=np.mean([json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs])-np.mean([json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"]["MCC"] for s in sds])
        comp=lambda d:(d["PRECISION"]+d["RECALL"]+d["MCC"]+d["DICE"]+d["VOLSIM"]+(1-d["HD95"]))/6
        cx=np.mean([comp(json.load(open(f))["adjusted_div_present"]) for f in fs]); cg=np.mean([comp(json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"]) for s in sds])
        print(f"  {sp:5s} "+"  ".join(line)+f"  | 복합 {cx:.4f}({cx-cg:+.4f})")
        if tag=="s2cl" and dm<=0: ok=False
        if tag=="s4d1" and cx-cg<=0: ok=False
    print(f"판정: {'PASS' if ok else 'FAIL'}  (s4 는 복합점수 기준, s2 는 MCC 기준)")
PYEOF
echo "S2S4_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$E/chain_status.md"
