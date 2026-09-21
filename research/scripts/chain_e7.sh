#!/usr/bin/env bash
# E7 — E5 에서 되살아난 두 노선을 e2e 로 재확인한다 (2026-08-19, E1b 앞으로 당김).
#
# 지금 우리 성능을 실제로 올릴 가능성이 있는 **유일한** 후보다.
# E5(268병변 x 5시드 반복 CV) 재심 결과:
#   synth (C11 합성샘플)  top1 +0.0410 (t=7.8) · macroMCC +0.0785   ← 압도적
#   geo   (C15 측지거리)  top1 +0.0164 (t=2.6)
# 둘 다 원래 기각했던 노선인데, 그 기각이 전부 단발 실행 판정이었다.
#
# CV 통과만으로 부족한 이유:
#   1) E5 에는 검출 위양성이 없다. 합성샘플은 사전확률을 크게 바꾸므로 β 와 상호작용한다.
#      -> val 에서 β 를 같이 스윕한다. 기존 β=0.5 가 합성 후에도 최적이란 보장이 없다.
#   2) E5 피처는 참조혈관 기반이다. 측지거리는 혈관이 완벽할 때의 값이라
#      예측혈관에서 남는지 별도로 봐야 한다.
#
# **판정선**: 현행(rf β0.5 τ0.5)의 시드 5판 평균 0.3477 ± 0.0199 를 넘어야 한다.
# 그래서 test 는 반드시 **시드 3판**으로 돌린다 — 단발 승리는 오늘 이미 한 번 속았다(C36).
# 설정 고정: model=rf, τ=0.5 (E4 에서 t=12.89 로 검증된 유일한 생존 결정).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"; GEOFEAT="$A/c34_feat_train_geoarc.json"
SYNTH="$A/c11_synth_rows.json"
log(){ echo "[e7 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=split $2=tag $3=beta $4=seed $5...=extra
  local sp=$1 tag=$2 bt=$3 sd=$4; shift 4
  local VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && { log "  $tag 있음"; return; }
  log "  $sp $tag (β=$bt seed=$sd)"
  CLF_SEED=$sd $PY -u c5_location_v2.py eval --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
    --model rf --use-pos --beta "$bt" --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$tag" "$@" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

# ── STEP1: val 에서 합성샘플용 β 만 고른다 (하이퍼파라미터 선택) ───────
log "=== STEP1: val — 합성샘플 β 스윕 ==="
for B in 0.0 0.25 0.5 0.75; do
  ev val "e7_synth_b$B" "$B" 0 --train-feat "$FEAT" --synth-feat "$SYNTH" --synth-repeat 2
done
BB=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
best=("0.5",-1)
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_e7_synth_b*.json")):
    m=re.search(r"_b([0-9.]+)\.json$",f)
    o=json.load(open(f)).get("official_div52") or {}
    if m and o.get("MCC",-1)>best[1]: best=(m.group(1),o["MCC"])
print(best[0])
PYEOF
)
log "=== 합성샘플 최적 β = $BB ==="

# ── STEP2: test — 3후보 x 시드 3판 (기준선은 E4 의 rf β0.5 τ0.5 5판) ──
log "=== STEP2: test (시드 3판) ==="
for SD in 0 1 2; do
  ev test "e7_synth_s${SD}"  "$BB" "$SD" --train-feat "$FEAT"    --synth-feat "$SYNTH" --synth-repeat 2
  ev test "e7_geo_s${SD}"    0.5   "$SD" --train-feat "$GEOFEAT" --use-geo
  ev test "e7_geosyn_s${SD}" "$BB" "$SD" --train-feat "$GEOFEAT" --use-geo --synth-feat "$SYNTH" --synth-repeat 2
done

log "=== STEP3: 요약 — 현행 시드산포와 같은 표에서 비교 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,re,sys,collections
import numpy as np
A=sys.argv[1]
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
g=collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(A,"c5_eval_test_e7_*_s?.json"))
               +glob.glob(os.path.join(A,"c5_eval_test_e4_rf_s?.json"))):
    m=re.search(r"c5_eval_test_(.+)_s(\d)\.json$",f)
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    g[m.group(1)].append((o['MCC'],c['MCC'],comp(c)))
print(f"\n{'설정':<14}{'n':>3}| {'cov.MCC 평균':>13}{'±':>8}{'범위':>19}| {'off.MCC':>9}")
rows=sorted(g.items(), key=lambda kv:-np.mean([r[1] for r in kv[1]]))
for k,v in rows:
    a=np.array(v)
    lab = k+" (현행 기준선)" if k=="e4_rf" else k
    print(f"{lab:<14}{len(a):>3}| {a[:,1].mean():>13.4f}{a[:,1].std():>8.4f}"
          f"{f'[{a[:,1].min():.4f},{a[:,1].max():.4f}]':>19}| {a[:,0].mean():>9.4f}")
if "e4_rf" in g:
    b=np.array([r[1] for r in g["e4_rf"]])
    print(f"\n  기준선 rf β0.5 τ0.5 = {b.mean():.4f} ± {b.std():.4f} (5판)")
    for k,v in rows:
        if k=="e4_rf": continue
        a=np.array([r[1] for r in v]); d=a.mean()-b.mean()
        se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
        t=d/se if se>0 else 0
        v_="★ 실재하는 개선" if d>0 and abs(t)>2.5 else ("열세" if d<0 and abs(t)>2.5 else "차이 없음")
        print(f"    {k:<14} Δ {d:+.4f}  t={t:+.2f}  {v_}")
    print("\n  ⚠️ 시드 3판이라 t 는 참고값이다. 통과하면 5판으로 확정한다.")
PYEOF
log "=== 완료 ==="
