#!/usr/bin/env bash
# C37 — ExtraTrees 에 맞춰 β/τ 재선택 (2026-08-18). train 268 학습, val 선택, test 확인.
# β=0.5·τ=0.5 는 RandomForest 에 맞춰 고른 값이다. C36 에서 모델이 ET 로 바뀌었으니
# 결정규칙도 다시 골라야 한다. ET 는 분할점이 무작위라 확률이 RF 보다 부드럽게 나오므로
# 확신 게이트 임계 τ 의 최적점이 옮겨갔을 가능성이 크다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[c37 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=split $2=tag $3=beta $4=tau
  local sp=$1 tag=$2 bt=$3 tau=$4 VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag (β=$bt τ=$tau)"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
    --model et --use-pos --beta "$bt" --conf-tau "$tau" --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

log "=== STEP1: val — ET β/τ 격자 ==="
for B in 0.25 0.5 0.75; do
  for T in 0.0 0.4 0.6; do ev val "c37_b${B}_t${T}" "$B" "$T"; done
done

BEST=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c37_b*.json")):
    m=re.search(r"c37_b([0-9.]+)_t([0-9.]+)\.json$",f)
    o=json.load(open(f)).get("official_div52") or {}
    if m and o: rows.append((o.get("MCC",-1),m.group(1),m.group(2)))
# 현행 (β=0.5, τ=0.5) 도 후보에 넣는다
c=os.path.join(sys.argv[1],"c5_eval_val_c36_et.json")
if os.path.exists(c):
    o=json.load(open(c)).get("official_div52") or {}
    rows.append((o.get("MCC",-1),"0.5","0.5"))
rows.sort(reverse=True)
for v,b,t in rows: print(f"  β={b} τ={t}  {v:.4f}", file=sys.stderr)
print(f"{rows[0][1]} {rows[0][2]}" if rows else "0.5 0.5")
PYEOF
)
BB=$(echo $BEST|cut -d' ' -f1); TT=$(echo $BEST|cut -d' ' -f2)
log "=== val 최적: β=$BB τ=$TT ==="

log "=== STEP2: test ==="
if [ "$BB" = "0.5" ] && [ "$TT" = "0.5" ]; then
  log "  현행 (β=0.5 τ=0.5) 이 그대로 최적 — 이미 잰 c36_et 가 결과"
else
  ev test "c37_b${BB}_t${TT}" "$BB" "$TT"
fi

log "=== STEP3: 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c3[67]*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c31_t0.5_b0.5.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
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
