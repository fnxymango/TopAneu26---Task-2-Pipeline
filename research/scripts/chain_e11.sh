#!/usr/bin/env bash
# E11 — E10(예측혈관 학습, +0.0393 t=4.68 5/5) 을 두 각도로 검증한다 (2026-08-19).
#
# 왜 더 재나: 오늘 세 번 속았다 — C36 ET(단발 +0.0303 → 5판 -0.0300),
# C11 synth(CV t=7.8 → e2e -0.0205), C15 geo(3판 t=2.02 → 5판 t=0.53).
# 게다가 두 학습 피처의 총량 통계가 거의 같은데(관측혈관 5.3 vs 5.5, 분기점 26.2 vs 26.8)
# +0.039 가 나왔다. 원인이 미묘해서 **원인을 모른 채로는 채택하지 않는다**.
#
#   (A) val 5판  — test 에서만 나는 현상인지. split 이 다르면 노이즈일 확률이 확 준다.
#   (B) 블록 분해 — 거리36 / 중첩36 / 분기점34 중 어느 블록을 예측혈관으로 바꿔야 이득인가.
#       전부 바꿔서 오른 건지 한 블록 때문인지 갈라야 원인을 안다.
#       하이브리드 피처를 만들어(참조판 + 해당 블록만 예측판) 각각 5판씩 잰다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REF="$A/c10_feat_train.json"; PRD="$A/c10_feat_train_predves.json"
log(){ echo "[e11 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── (B) 하이브리드 피처 생성 ──────────────────────────────────────────
log "=== 하이브리드 피처 생성 (참조판 + 지정 블록만 예측판) ==="
$PY - "$REF" "$PRD" "$A" <<'PYEOF'
import json,sys,os
ref=json.load(open(sys.argv[1])); prd=json.load(open(sys.argv[2])); A=sys.argv[3]
key=lambda r:(r["case"],r["lesion_mask_idx"])
pm={key(r):r for r in prd}
BLOCKS={"dist":["dist_mm"],"ov":["overlap"],"bp":["bp_mm"],
        "distov":["dist_mm","overlap"]}
for name,flds in BLOCKS.items():
    out=[]
    for r in ref:
        q=pm.get(key(r))
        s=dict(r)
        if q:
            for f in flds: s[f]=q[f]
        out.append(s)
    p=os.path.join(A,f"e11_feat_hyb_{name}.json")
    json.dump(out,open(p,"w"),ensure_ascii=False)
    n=sum(1 for r,s in zip(ref,out) if any(r[f]!=s[f] for f in flds))
    print(f"  {name:<8} {flds} · 값이 바뀐 병변 {n}/{len(ref)} -> {os.path.basename(p)}")
PYEOF

ev(){ # $1=split $2=tag $3=feat
  local sp=$1 tag=$2 feat=$3 VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  for SD in 0 1 2 3 4; do
    local T="${tag}_s${SD}"
    [ -f "$A/c5_eval_${sp}_${T}.json" ] && continue
    log "  $sp $T"
    CLF_SEED=$SD $PY -u c5_location_v2.py eval --train-feat "$feat" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "$T" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
  done
}

# ── (A) val 5판: 현행 vs 예측혈관 ─────────────────────────────────────
log "=== (A) val 5판 ==="
ev val e11_val_ref  "$REF"
ev val e11_val_prd  "$PRD"

# ── (B) test 블록 분해 5판 ────────────────────────────────────────────
log "=== (B) test 블록 분해 5판 ==="
for B in dist ov bp distov; do ev test "e11_hyb_$B" "$A/e11_feat_hyb_$B.json"; done

log "=== 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def load(sp,tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_{sp}_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c,ofc=d.get("adjusted_div_present"),d.get("official_div52")
        if c and ofc: o[sd]=(c["MCC"],ofc["MCC"])
    return o
def cmp(g,b,lab,w=22):
    sds=sorted(set(g)&set(b))
    if len(sds)<2: print(f"  {lab:<{w}} (n={len(sds)}) 데이터 부족"); return
    ga=np.array([g[s][0] for s in sds]); ba=np.array([b[s][0] for s in sds]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0
    v="★" if d.mean()>0 and t>2.5 else ("↓" if d.mean()<0 and t<-2.5 else " ")
    print(f"  {lab:<{w}}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}"
          f"{d.mean():>+10.4f}{t:>+7.2f}{int((d>0).sum()):>4}/{len(d)}  {v}")

print("\n=== (A) val — test 와 같은 방향인가 ===")
print(f"  {'설정':<22}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ':>10}{'t':>7}{'승':>6}")
vr,vp=load("val","e11_val_ref"),load("val","e11_val_prd")
if vr:
    a=np.array([vr[s][0] for s in sorted(vr)])
    print(f"  {'참조혈관 학습(기준)':<22}{len(a):>3}{a.mean():>10.4f}{a.std():>8.4f}")
cmp(vp,vr,"예측혈관 학습")

print("\n=== (B) test 블록 분해 — 어느 블록이 이득을 내나 ===")
b=load("test","e4_rf")
a=np.array([b[s][0] for s in sorted(b)])
print(f"  {'설정':<22}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ':>10}{'t':>7}{'승':>6}")
print(f"  {'참조혈관 학습(기준)':<22}{len(a):>3}{a.mean():>10.4f}{a.std():>8.4f}")
for tag,lab in (("e11_hyb_dist","+거리36만 예측"),("e11_hyb_ov","+중첩36만 예측"),
                ("e11_hyb_bp","+분기점34만 예측"),("e11_hyb_distov","+거리·중첩 예측"),
                ("e10_predves","전부 예측 (E10)")):
    cmp(load("test",tag),b,lab)
print("\n  ★ = Δ>0 이고 t>2.5. 블록별 이득의 합이 전체와 맞는지 보면 원인이 갈린다.")
PYEOF
log "=== E1b 재개 ==="
setsid nohup env TOPANEU_ROOT="$R" bash "$S/chain_e1b.sh" >> "$E/e1b_chain.log" 2>&1 < /dev/null & disown
setsid nohup env TOPANEU_ROOT="$R" bash "$S/watchdog_e.sh" >> "$E/watchdog_e.log" 2>&1 < /dev/null & disown
log "=== 완료 ==="
