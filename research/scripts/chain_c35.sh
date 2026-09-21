#!/usr/bin/env bash
# C35 — geo 블록을 val e2e 로 확인하고, 이기면 test (2026-08-18).
# train 268 로만 학습(PROJECT_RULES.md §1, val 은 선택 전용). β=0.5 τ=0.5 고정, **피처만** 바꾼다.
# CV 이득이 +0.0066 (macroMCC 0.3493 -> 0.3558) 으로 노이즈 범위라 e2e 확인이 필요하다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
GA="$A/c34_feat_train_geoarc.json"
log(){ echo "[c35 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=split $2=tag $3=extra
  local sp=$1 tag=$2 ex=${3:-} VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag $ex"
  $PY -u c5_location_v2.py eval --train-feat "$GA" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 $ex \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

log "=== STEP1: val — geo 켜고/끄고 ==="
ev val c35_nogeo ""
ev val c35_geo   "--use-geo"

WIN=$($PY - "$A" <<'PYEOF'
import json,os,sys
A=sys.argv[1]
def mcc(t):
    p=os.path.join(A,f"c5_eval_val_{t}.json")
    return (json.load(open(p)).get("official_div52") or {}).get("MCC",-1) if os.path.exists(p) else -1
print("geo" if mcc("c35_geo")>mcc("c35_nogeo") else "none")
PYEOF
)
log "=== val 판정: $WIN ==="
if [ "$WIN" = "geo" ]; then
  log "=== STEP2: test 확인 ==="
  ev test c35_geo "--use-geo"
else
  log "geo 가 val e2e 에서 이기지 못함 — test 생략, C15 노선 최종 기각"
fi

log "=== STEP3: 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c3[1245]*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c26_b0.5.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f))
    if d.get("eval_mode")=="covered_gt": continue
    if "GT" in str(d.get("aneurysm_pred_dir","")): continue
    o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    tag=os.path.basename(f)[len("c5_eval_test_"):-5]
    if tag.startswith("c32"): tag+=" (철회: val학습)"
    rows.append((tag,o,c))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<26}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.HD95':>9}{'cov.복합':>9}")
for tag,o,c in rows:
    print(f"{tag:<26}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}")
PYEOF
log "=== 완료 ==="
