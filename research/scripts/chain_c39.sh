#!/usr/bin/env bash
# C39 — ExtraTrees 를 끝까지 짜낸다 (2026-08-18). train 268 학습 · val 선택 · test 확인.
#
# C36 에서 RF -> ET 만으로 test cov.MCC 0.3306 -> 0.3609 (+9.2%) 가 나왔는데
# 정작 ET 는 기본값으로만 썼다. 그리고 C38 에서 확신 게이트는 노이즈로 판명(0.3609 vs 0.3597).
# 남은 것은 '새 정보 넣기'가 아니라 **이미 가진 것을 덜 낭비하기** 세 가지다.
#
#   STEP1  ET 기준 분류 천장 재측정 — 남은 여지가 분류 쪽인지 검출 쪽인지 확정
#   STEP2  ET 하이퍼파라미터 (max_features · min_samples_leaf)
#          기본 'sqrt' 는 112차원에서 분할마다 10개만 본다 — 이 표본수에선 너무 적을 수 있다
#   STEP3  추론 시 미러 TTA — 미러를 학습에만 쓰고 추론엔 안 썼다. 좌우대칭은 이미 성립이
#          확인된 사실(증강 채택 근거)이라, 새 정보 없이 분산만 줄이는 공짜 수다
#   STEP4  val 최고 하나만 test
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[c39 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=split $2=tag $3=extra flags ; ET_* 는 환경변수로 넘어온다
  local sp=$1 tag=$2 ex=${3:-} VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="--aneurysm-pred-dir $P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="--aneurysm-pred-dir $P/aneu_test_probavgf"; fi
  [ "${CEIL:-0}" = 1 ] && AD=""
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag  [mf=${ET_MAX_FEATURES:-sqrt} leaf=${ET_MIN_LEAF:-1}] $ex"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" $AD \
    --model et --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 $ex \
    --tag "$tag" 2>&1 | grep -E '"MCC"|"top1_accuracy"' | head -2 || log "    실패"
}

log "=== STEP1: ET 기준 분류 천장 (test, GT 병변) ==="
CEIL=1 ev test c39_ceil_et ""
CEIL=0

log "=== STEP2: val — ET max_features x min_samples_leaf ==="
for MF in sqrt 0.2 0.4 0.7; do
  for ML in 1 2; do
    ET_MAX_FEATURES=$MF ET_MIN_LEAF=$ML ev val "c39_mf${MF}_ml${ML}" ""
  done
done

log "=== STEP3: val — 추론 미러 TTA ==="
ev val c39_tta "--tta"

log "=== val 순위 ==="
BEST=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c39_*.json"))+[
         os.path.join(sys.argv[1],"c5_eval_val_c36_et.json")]:
    if not os.path.exists(f): continue
    o=json.load(open(f)).get("official_div52") or {}
    n=os.path.basename(f)[len("c5_eval_val_"):-5]
    if o: rows.append((o.get("MCC",-1),n))
rows.sort(reverse=True)
for v,n in rows: print(f"  {n:<22} {v:.4f}", file=sys.stderr)
print(rows[0][1] if rows else "c36_et")
PYEOF
)
log "=== val 최고: $BEST ==="

log "=== STEP4: test 확인 ==="
case "$BEST" in
  c36_et) log "  기본 ET 가 그대로 최고 — 이미 잰 c36_et 가 결과" ;;
  c39_tta) ev test c39_tta_test "--tta" ;;
  c39_mf*) MF=$(echo $BEST|sed 's/c39_mf\(.*\)_ml.*/\1/'); ML=$(echo $BEST|sed 's/.*_ml//')
           ET_MAX_FEATURES=$MF ET_MIN_LEAF=$ML ev test "${BEST}_test" "" ;;
esac

log "=== 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c3[6789]*.json"))):
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    ceil=" (천장)" if "GT" in str(d.get("aneurysm_pred_dir","")) else ""
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5]+ceil,o,c,d.get("top1_accuracy")))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<22}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.HD95':>9}{'cov.복합':>9}{'top1':>8}")
for tag,o,c,t1 in rows:
    print(f"{tag:<22}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}{(t1 or 0):>8.3f}")
PYEOF
log "=== 완료 ==="
