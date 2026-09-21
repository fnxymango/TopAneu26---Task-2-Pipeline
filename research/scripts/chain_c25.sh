#!/usr/bin/env bash
# C25 — 크롭 분기를 MIP+RF 로 교체하고 e2e 까지 (2026-08-17). **한 스크립트**로 묶어 경계에서 안 끊기게.
#   STEP1  크롭 MIP+RF CV + 앙상블 스윕, val/test 크롭에 확률 내보내기
#   STEP2  val e2e — w 스윕. **official 과 covered-GT 를 매번 둘 다** 낸다(사용자 지시 2026-08-17).
#   STEP3  val MCC 최고 w 로 test — official + covered-GT
#   STEP4  요약표
# 대기 조건에 pgrep 를 쓰지 않는다(앞서 두 번 정체시킨 원인). 산출물 존재로 판정.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"; CR="$E/_c24_crops"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"; BEST="--model rf --beta 1.0 --use-pos"
log(){ echo "[c25 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── STEP1: 크롭 분기 학습 + 확률 내보내기 ────────────────────────────
if [ ! -f "$A/c25_prob_test_det.json" ]; then
  log "=== STEP1: 크롭 MIP+RF CV + 확률 내보내기 ==="
  $PY -u c25_crop_mip.py --crop-dir "$CR/train" --feat "$FEAT" --beta 1.0 \
      --apply val_gt="$CR/val_gt"   --apply val_det="$CR/val_det" \
      --apply test_gt="$CR/test_gt" --apply test_det="$CR/test_det" 2>&1 || { log "STEP1 실패"; exit 1; }
else
  log "STEP1 이미 완료 — 건너뜀"
fi

# ── 평가 헬퍼: official 과 covered-GT 를 한 쌍으로 ────────────────────
ev(){ # $1=split $2=tag $3=w $4=crop_prob(det용) $5=crop_prob(gt용)
  local sp=$1 tag=$2 w=$3 pd=${4:-} pg=${5:-}
  local VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD=aneu_val_probavgf
  else VD=vespp_test; VB=vespp_test; AD=aneu_test_probavgf; fi
  local EX="" EXG=""
  [ -n "$pd" ] && EX="--crop-prob $pd --crop-w $w"
  [ -n "$pg" ] && EXG="--crop-prob $pg --crop-w $w"
  if [ ! -f "$A/c5_eval_${sp}_${tag}.json" ]; then
    log "  official  $sp $tag"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/$AD" \
      $BEST $EX --tag "$tag" 2>&1 | grep -E '"MCC"|"top1_accuracy"' | head -2 || log "    실패"
  fi
  if [ ! -f "$A/c5_eval_${sp}_${tag}_cov.json" ]; then
    log "  coveredGT $sp $tag"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/$AD" --covered-gt \
      $BEST $EXG --tag "${tag}_cov" 2>&1 | grep -E '"MCC"|"top1_accuracy"|"detect_coverage"' | head -3 || log "    실패"
  fi
}

log "=== STEP2: val — w 스윕 (official + covered-GT) ==="
ev val c25_w0.0 0.0
for W in 0.2 0.3 0.5; do
  ev val "c25_w$W" "$W" "$A/c25_prob_val_det.json" "$A/c25_prob_val_gt.json"
done

# ── STEP3: val MCC 최고 w 를 test 에 ──────────────────────────────────
BESTW=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
best=(None,-1)
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c25_w*.json")):
    if f.endswith("_cov.json"): continue
    m=re.search(r"c25_w([0-9.]+)\.json$",f)
    if not m: continue
    d=json.load(open(f)); o=d.get("official_div52") or {}
    if o.get("MCC",-1)>best[1]: best=(m.group(1),o["MCC"])
print(best[0] or "0.0")
PYEOF
)
log "=== STEP3: test — val 최고 w=$BESTW ==="
if [ "$BESTW" = "0.0" ]; then
  ev test c25_w0.0 0.0
else
  ev test c25_w0.0 0.0
  ev test "c25_w$BESTW" "$BESTW" "$A/c25_prob_test_det.json" "$A/c25_prob_test_gt.json"
fi

# ── STEP4: 요약 ───────────────────────────────────────────────────────
log "=== STEP4: 요약 (official / covered-GT) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
def comp(o): return (sum(o[k] for k in K)+1-o['HD95'])/6
for sp in ("val","test"):
    rows=[]
    for f in sorted(glob.glob(os.path.join(sys.argv[1],f"c5_eval_{sp}_c25_w*.json"))):
        if f.endswith("_cov.json"): continue
        tag=os.path.basename(f)[len(f"c5_eval_{sp}_"):-5]
        d=json.load(open(f)); o=d.get("official_div52")
        cf=f[:-5]+"_cov.json"
        c=json.load(open(cf)).get("official_div52") if os.path.exists(cf) else None
        cd=json.load(open(cf)) if os.path.exists(cf) else {}
        if not o: continue
        rows.append((tag,o,c,d.get("top1_accuracy"),cd.get("top1_accuracy"),cd.get("detect_coverage")))
    if not rows: continue
    print(f"\n===== {sp} =====")
    print(f"{'설정':<12}{'off.MCC':>9}{'off.DICE':>10}{'off.복합':>10}"
          f"{'cov.MCC':>10}{'cov.DICE':>10}{'cov.복합':>10}{'cov.top1':>10}{'검출커버':>9}")
    for tag,o,c,t1,ct1,cov in sorted(rows,key=lambda r:-r[1]['MCC']):
        cs=f"{c['MCC']:>10.4f}{c['DICE']:>10.4f}{comp(c):>10.4f}" if c else f"{'-':>10}{'-':>10}{'-':>10}"
        print(f"{tag:<12}{o['MCC']:>9.4f}{o['DICE']:>10.4f}{comp(o):>10.4f}{cs}"
              f"{(ct1 if ct1 is not None else 0):>10.3f}{(cov if cov is not None else 0):>9.3f}")
PYEOF
log "=== 완료 ==="
