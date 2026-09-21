#!/usr/bin/env bash
# E17 — 중첩만-예측(e11_hyb_ov) 의 val 5시드. E15p 에서 292 OOF 가 ★채택(t=3.77 5/5)을
# 냈고 test 도 ★(t=4.34 5/5)인데, val 만 '전부 예측'판(-0.0171)밖에 없다.
# 채택 문서에 val 칸을 비워두지 않으려고 같은 축으로 5판 채운다. 판정은 이미 292 로 났다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
log(){ echo "[e17 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
PIDS=()
for SD in 0 1 2 3 4; do
  T="e17_val_ov_s${SD}"
  [ -f "$A/c5_eval_val_${T}.json" ] && { log "  $T 이미 있음"; continue; }
  CLF_SEED=$SD nohup $PY -u c5_location_v2.py eval \
    --train-feat "$A/e11_feat_hyb_ov.json" --split val \
    --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" \
    --aneurysm-pred-dir "$P/aneu_val_probavgf" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$T" > "$E/e17_${T}.log" 2>&1 &
  PIDS+=($!); log "  띄움 $T (pid $!)"
done
for p in "${PIDS[@]}"; do wait "$p"; done
log "=== 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_val_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c=d.get("adjusted_div_present"); ofc=d.get("official_div52")
        if c and ofc: o[sd]=(c["MCC"],ofc["MCC"])
    return o
ref,prd,ov=load("e11_val_ref"),load("e11_val_prd"),load("e17_val_ov")
a=np.array([ref[s][0] for s in sorted(ref)])
print(f"\n[val 42]  {'설정':<18}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ':>10}{'t':>7}{'승':>6}")
print(f"          {'참조혈관(기준)':<18}{len(a):>3}{a.mean():>10.4f}{a.std():>8.4f}")
for g,lab in ((prd,"전부 예측(기각)"),(ov,"중첩만 예측 ★")):
    sds=sorted(set(g)&set(ref))
    if len(sds)<2: print(f"          {lab:<18} 데이터 부족 n={len(sds)}"); continue
    ga=np.array([g[s][0] for s in sds]); ba=np.array([ref[s][0] for s in sds]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0
    print(f"          {lab:<18}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}"
          f"{d.mean():>+10.4f}{t:>+7.2f}{int((d>0).sum()):>4}/{len(d)}")
    print(f"          {'':<18}   시드별 Δ {[round(x,4) for x in d]}")
PYEOF
log "=== 완료 ==="
