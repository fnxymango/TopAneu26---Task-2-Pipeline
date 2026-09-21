#!/usr/bin/env bash
# C36 — 분류기 모델 교체 (2026-08-18). train 268 학습, val 로 선택, test 로 확인.
#
# 지금까지 RF 하나만 썼다. C21 에서 GBM/앙상블을 시험해 기각했지만 그건
# **β 이중보정(β=1.0) 상태에서 train CV 로** 판정한 것이다. 오늘 같은 조건에서
# C15 측지 노선의 부호가 뒤집혔었고(CV +0.0066), 다시 val e2e 에서 뒤집혔다(-0.057).
# 그래서 (1) 교정된 설정(β=0.5, τ=0.5)에서 (2) **val e2e** 로 다시 잰다.
#
# 후보: rf(현행) · et(ExtraTrees) · gb(HistGB) · logreg · mlp · rf+et · rf+gb · rf+et+gb
# ExtraTrees 를 넣는 이유 — 268샘플 x 112차원처럼 표본이 적고 차원이 높으면
# 분할점을 무작위로 뽑는 쪽이 분산이 낮고, RF 와 같은 계열이라 앙상블 상보성도 있다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[c36 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=split $2=model
  local sp=$1 m=$2 VD VB AD tag="c36_$2"
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $m"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
    --model "$m" --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

log "=== STEP1: val — 분류기 8종 ==="
for M in rf et gb logreg mlp rf_et rf_gb rf_et_gb; do ev val "$M"; done

log "=== val 순위 (MCC) ==="
BEST=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c36_*.json")):
    m=re.search(r"c36_(.+)\.json$",f)
    o=json.load(open(f)).get("official_div52") or {}
    if m and o: rows.append((o.get("MCC",-1),m.group(1)))
rows.sort(reverse=True)
for v,n in rows: print(f"  {n:<10} {v:.4f}", file=sys.stderr)
print(rows[0][1] if rows else "rf")
PYEOF
)
log "=== val 최고 모델: $BEST ==="

log "=== STEP2: test 확인 (현행 rf 와 함께) ==="
ev test rf
[ "$BEST" != "rf" ] && ev test "$BEST"

log "=== STEP3: 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c36_*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c31_t0.5_b0.5.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f))
    o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5],o,c))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<20}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.HD95':>9}{'cov.복합':>9}")
for tag,o,c in rows:
    print(f"{tag:<20}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}")
PYEOF
log "=== 완료 ==="
