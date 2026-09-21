#!/bin/bash
# 마스터 체인 (2026-08-26 16:20 KST) — 끊김 없이: surx5 판정 -> N1 판정 -> N2 e2e -> 통과조합 최종런.
# 모든 판정 결과는 experiments/chain_status.md 에 append. 규칙은 사전 고정:
#   채택 = 현행 최고(X5+gC: test 0.3972 / val 0.4725) 대비 test Δ>0 AND val Δ>0.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
ST=$E/chain_status.md
log(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*" | tee -a "$ST"; }

judge(){ # tag_prefix 라벨  -> x5g 대비 판정 append
  $PY - "$1" "$2" <<'PYEOF' >> "$ST" 2>&1
import json,glob,numpy as np,sys
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
tag,lab=sys.argv[1],sys.argv[2]
X5={"test":"c5_eval_test_x5g_test_s%d.json","val":"c5_eval_val_x5g_val_s%d.json"}
ok=True; lines=[]
for sp in ("test","val"):
    fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_{tag}_{sp}_s*.json"))
    if not fs: lines.append(f"  {sp}: 결과 없음"); ok=False; continue
    sds=[int(f.split("_s")[-1][:-5]) for f in fs]
    x=np.array([json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs])
    g=np.array([json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"]["MCC"] for s in sds])
    d=x-g; sd=d.std(ddof=1) if len(d)>1 else 0
    t=d.mean()/(sd/np.sqrt(len(d))) if sd>1e-12 else float("nan")
    lines.append(f"  {sp:5s} {x.mean():.4f}±{x.std(ddof=1):.4f}  vs x5g {g.mean():.4f}  Δ{d.mean():+.4f} t{t:+.2f} {int((d>0).sum())}/{len(d)}")
    if d.mean()<=0: ok=False
print(f"\n### {lab} ({tag})")
print("\n".join(lines))
print(f"판정: {'PASS' if ok else 'FAIL'}")
open(A+f"/chain_{tag}.verdict","w").write("PASS" if ok else "FAIL")
PYEOF
}

run_eval(){ # split aneu_suffix seed tag extra_env...
  local sp=$1 an=$2 sd=$3 tg=$4; shift 4
  local ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  env "$@" CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${an}" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tg}_s${sd}" > "$E/${tg}_s${sd}.log" 2>&1
}

log "체인 시작"
# ── A. surx5 판정 ─────────────────────────────────────────────
until grep -q "^DONE" "$E/q15_surx5.log" 2>/dev/null; do sleep 30; done
judge surx "②' 수술적(X5기반)+gC"
log "A 완료 (surx)"

# ── B. N1 판정 (q16 은 q15 뒤 자동 시작됨) ──────────────────────
until grep -q "^DONE" "$E/q16_n1.log" 2>/dev/null; do sleep 30; done
judge n1a "N1 p2>0.20"
judge n1b "N1 p2>0.25"
log "B 완료 (N1)"

# ── C. N2 e2e (스크리닝 결과와 무관하게 실측 — 판정은 e2e 로) ────
until [ -f "$A/case_vessel_vols.npz" ]; do sleep 20; done
$PY -u p_cprior_table.py >> "$ST" 2>&1 || { log "C 실패: prior 테이블"; exit 1; }
# w 는 OOF 스크리닝 최적을 쓰되 로그에서 자동 추출 실패시 0.5
W=$($PY - <<'PYEOF'
import re
best,w=None,0.5
try:
    txt=open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/q17_n2.log").read()
    blk=txt.split("[292 OOF]")[1].split("[")[0]
    for m in re.finditer(r"w=([\d.]+)\s+top1 [\d.]+\s+macroRec [\d.]+\s+MCC ([\d.]+)",blk):
        if best is None or float(m.group(2))>best: best,w=float(m.group(2)),float(m.group(1))
except Exception: pass
print(w)
PYEOF
)
log "C: N2 e2e 시작 (w=$W, X5+gC 기반)"
for sd in 0 1 2 3 4; do
  run_eval test P55ff $sd n2_test TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_CPRIOR="$A/case_prior.json" TOPANEU_CPRIOR_W=$W &
  run_eval val P55ff $sd n2_val TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_CPRIOR="$A/case_prior.json" TOPANEU_CPRIOR_W=$W &
done
wait
judge n2 "N2 케이스prior (w=$W, gC 위)"
log "C 완료 (N2)"

# ── D. 통과 조합 최종런 ────────────────────────────────────────
# 슬리버 게이트: n1a/n1b 중 PASS 이며 Δ 큰 쪽, 없으면 gC 유지.
# n2 PASS 면 prior 추가.
SLIV="gC"
[ "$(cat $A/chain_n1a.verdict 2>/dev/null)" = PASS ] && SLIV="n1a"
if [ "$(cat $A/chain_n1b.verdict 2>/dev/null)" = PASS ]; then
  [ "$SLIV" = n1a ] && SLIV=$($PY - <<'PYEOF'
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
def m(t):
    v=[]
    for sp in ("test","val"):
        fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_{t}_{sp}_s*.json"))
        v.append(np.mean([json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs]))
    return sum(v)
print("n1a" if m("n1a_test".replace("_test",""))>=m("n1b") else "n1b")
PYEOF
) || SLIV="n1b"
fi
N2ON=$([ "$(cat $A/chain_n2.verdict 2>/dev/null)" = PASS ] && echo 1 || echo 0)
log "D: 최종조합 = 슬리버:$SLIV · N2:$N2ON"
if [ "$SLIV" = gC ] && [ "$N2ON" = 0 ]; then
  log "D: 조합이 현행(X5+gC)과 동일 — 추가런 불필요. 최종 = X5+gC"
else
  EX=""
  case $SLIV in
    gC)  EX="TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7" ;;
    n1a) EX="TOPANEU_TOPK=2 TOPANEU_TOPK_P2=0.20" ;;
    n1b) EX="TOPANEU_TOPK=2 TOPANEU_TOPK_P2=0.25" ;;
  esac
  [ "$N2ON" = 1 ] && EX="$EX TOPANEU_CPRIOR=$A/case_prior.json TOPANEU_CPRIOR_W=$W"
  for sd in 0 1 2 3 4; do
    run_eval test P55ff $sd fincombo_test $EX &
    run_eval val P55ff $sd fincombo_val $EX &
  done
  wait
  judge fincombo "최종조합 ($SLIV + N2:$N2ON)"
fi
log "DONE_ALL"
