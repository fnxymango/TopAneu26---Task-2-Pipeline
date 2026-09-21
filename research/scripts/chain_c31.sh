#!/usr/bin/env bash
# C31 — β 를 확신이 낮은 병변에만 건다 (2026-08-17). **한 스크립트**로.
#
# 오늘 밝혀낸 것: β 의 가치는 "희소 클래스를 살리는 것"이 아니라
# **"위양성을 무해한 곳에 버리는 것"** 이다. GT 에 없는 클래스는 tp=fn=0 이라
# MCC 분모가 0 -> 0 그대로여서 벌점이 없다. 그래서
#   GT 병변만 보는 train CV (FP 없음)  -> β=0 이 최선  (macroMCC 0.3284 vs 0.3106)
#   FP 가 섞이는 val e2e               -> β=0.5 가 최선 (MCC 0.2801 vs 0.2233)
# 로 정반대 결론이 나왔다. 둘을 동시에 만족시키려면 확신이 높은 병변(진짜 TP 후보)은
# argmax 로 두고 낮은 것만 희소 쪽으로 밀면 된다.
#
# 확신 분포(train CV, GT 병변 268개):
#   τ=0.3  저확신 11%  정확도 0.207 / 고확신 0.720
#   τ=0.4  저확신 26%  정확도 0.394 / 고확신 0.761
#   τ=0.5  저확신 42%  정확도 0.455 / 고확신 0.814
#
# 선택은 val, 보고는 test. 표는 PROJECT_RULES.md 6-1 대로 official_div52 + covered_gt(÷36),
# 6지표 전부, 순위는 MCC.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[c31 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=split $2=tag $3=beta(저확신) $4=tau $5=beta_hi
  local sp=$1 tag=$2 bt=$3 tau=$4 bh=$5 VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag (β=$bt τ=$tau β_hi=$bh)"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
    --model rf --use-pos --beta "$bt" --conf-tau "$tau" --conf-beta-hi "$bh" \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

log "=== STEP1: val — 확신 게이트 격자 ==="
for TAU in 0.3 0.4 0.5 0.6; do
  for B in 0.5 1.0; do
    ev val "c31_t${TAU}_b${B}" "$B" "$TAU" 0.0
  done
done

BEST=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
best=(None,-1)
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c31_*.json")):
    m=re.search(r"c31_t([0-9.]+)_b([0-9.]+)\.json$",f)
    if not m: continue
    o=json.load(open(f)).get("official_div52") or {}
    if o.get("MCC",-1)>best[1]: best=((m.group(1),m.group(2)),o["MCC"])
b=os.path.join(sys.argv[1],"c5_eval_val_c26_b0.5.json")     # 현행 최고 (τ 없음)
bm=(json.load(open(b)).get("official_div52") or {}).get("MCC",0) if os.path.exists(b) else 0
print(f"{best[0][0]} {best[0][1]}" if best[0] and best[1]>bm else "none none")
PYEOF
)
BT=$(echo $BEST | cut -d' ' -f1); BB=$(echo $BEST | cut -d' ' -f2)
log "=== val 최적: τ=$BT β=$BB (현행 β=0.5 대비) ==="

log "=== STEP2: test ==="
if [ "$BT" != "none" ]; then
  ev test "c31_t${BT}_b${BB}" "$BB" "$BT" 0.0
else
  log "확신 게이트가 현행보다 낫지 않음 — test 생략"
fi

log "=== STEP3: test 요약 (MCC 순, official_div52 + covered_gt ÷36) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c*.json"))):
    d=json.load(open(f))
    if d.get("eval_mode")=="covered_gt": continue
    if "GT" in str(d.get("aneurysm_pred_dir","")): continue      # 천장 제외
    o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5],o,c))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<18}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.VolSim':>11}{'cov.HD95':>9}{'cov.복합':>9}")
for tag,o,c in rows[:8]:
    print(f"{tag:<18}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['VOLSIM']:>11.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}")
PYEOF
log "=== 완료 ==="
