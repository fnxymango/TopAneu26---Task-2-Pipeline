#!/usr/bin/env bash
# C26 — β 이중보정 교정 + 크롭 혼합 (2026-08-17). **한 스크립트**로 묶어 경계에서 안 끊기게.
#
# 발견: 프로덕션 설정 --beta 1.0 이 이중 보정이었다. c5 fit_model 의 RF 는 이미
# class_weight="balanced" 로 클래스 균형을 맞추는데 그 위에 사전확률로 또 나눈다.
# train CV(268병변)에서 β=1.0 -> β=0.0 만으로 macroRec 0.324->0.372, top-1 0.496->0.668.
# 다만 그건 참조 혈관마스크 기준이고 실지표는 예측 혈관 e2e 라 여기서 다시 잰다.
#
#   STEP1  val β 스윕 (크롭 없음) — 이중보정 교정만의 효과
#   STEP2  val 최적 β 위에서 크롭 혼합 w 스윕
#   STEP3  val 최고(MCC)를 test 로
#   STEP4  요약
# 모든 평가는 official 과 covered-GT 를 **쌍으로** 낸다(사용자 지시 2026-08-17).
# 대기 조건에 pgrep 를 쓰지 않는다(앞서 두 번 정체시킨 원인).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[c26 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=split $2=tag $3=beta $4=w $5=crop여부(1이면 사용)
  local sp=$1 tag=$2 bt=$3 w=$4 usec=${5:-0}
  local VD VB AD; if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD=aneu_val_probavgf
  else VD=vespp_test; VB=vespp_test; AD=aneu_test_probavgf; fi
  local CD="" CG=""
  [ "$usec" = 1 ] && { CD="--crop-prob $A/c25_prob_${sp}_det.json --crop-w $w"
                       CG="--crop-prob $A/c25_prob_${sp}_gt.json  --crop-w $w"; }
  if [ ! -f "$A/c5_eval_${sp}_${tag}.json" ]; then
    log "  official  $sp $tag (β=$bt w=$w)"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/$AD" \
      --model rf --use-pos --beta "$bt" $CD --tag "$tag" 2>&1 \
      | grep -E '"MCC"|"top1_accuracy"' | head -2 || log "    실패"
  fi
  if [ ! -f "$A/c5_eval_${sp}_${tag}_cov.json" ]; then
    log "  coveredGT $sp $tag"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/$AD" --covered-gt \
      --model rf --use-pos --beta "$bt" $CG --tag "${tag}_cov" 2>&1 \
      | grep -E '"MCC"|"top1_accuracy"' | head -2 || log "    실패"
  fi
}

log "=== STEP1: val β 스윕 (이중보정 교정) ==="
for B in 0.0 0.25 0.5 0.75; do ev val "c26_b$B" "$B" 0 0; done
# β=1.0 은 c25_w0.0 으로 이미 있음 — 복사해 표에 함께 나오게 한다
for suf in "" "_cov"; do
  [ -f "$A/c5_eval_val_c25_w0.0${suf}.json" ] && cp -n "$A/c5_eval_val_c25_w0.0${suf}.json" \
      "$A/c5_eval_val_c26_b1.0${suf}.json"
done

BB=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
best=("1.0",-1)
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c26_b*.json")):
    if f.endswith("_cov.json"): continue
    m=re.search(r"c26_b([0-9.]+)\.json$",f)
    if not m: continue
    o=json.load(open(f)).get("official_div52") or {}
    if o.get("MCC",-1)>best[1]: best=(m.group(1),o["MCC"])
print(best[0])
PYEOF
)
log "=== STEP2: val 최적 β=$BB 위에서 크롭 혼합 w 스윕 ==="
for W in 0.3 0.5 0.7; do ev val "c26_b${BB}_w$W" "$BB" "$W" 1; done

BW=$($PY - "$A" "$BB" <<'PYEOF'
import json,glob,os,sys,re
A,bb=sys.argv[1],sys.argv[2]
best=("0",-1)
for f in glob.glob(os.path.join(A,f"c5_eval_val_c26_b{bb}_w*.json")):
    if f.endswith("_cov.json"): continue
    m=re.search(r"_w([0-9.]+)\.json$",f)
    if not m: continue
    o=json.load(open(f)).get("official_div52") or {}
    if o.get("MCC",-1)>best[1]: best=(m.group(1),o["MCC"])
b0=json.load(open(os.path.join(A,f"c5_eval_val_c26_b{bb}.json"))).get("official_div52") or {}
print(best[0] if best[1]>b0.get("MCC",0) else "0")
PYEOF
)
log "=== STEP3: test — β=$BB, 크롭 w=$BW ==="
ev test "c26_b$BB" "$BB" 0 0
[ "$BW" != "0" ] && ev test "c26_b${BB}_w$BW" "$BB" "$BW" 1
# 기존 채택 설정(β=1.0)도 test 에서 같은 표에 놓고 비교
ev test "c26_b1.0" 1.0 0 0

log "=== STEP4: 요약 (official / covered-GT) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
def comp(o): return (sum(o[k] for k in K)+1-o['HD95'])/6
for sp in ("val","test"):
    rows=[]
    for f in sorted(glob.glob(os.path.join(sys.argv[1],f"c5_eval_{sp}_c26_*.json"))):
        if f.endswith("_cov.json"): continue
        tag=os.path.basename(f)[len(f"c5_eval_{sp}_"):-5]
        d=json.load(open(f)); o=d.get("official_div52")
        cf=f[:-5]+"_cov.json"; cd=json.load(open(cf)) if os.path.exists(cf) else {}
        c=cd.get("official_div52")
        if not o: continue
        rows.append((tag,o,c,d.get("top1_accuracy"),cd.get("top1_accuracy"),cd.get("detect_coverage")))
    if not rows: continue
    print(f"\n===== {sp} =====")
    print(f"{'설정':<18}{'off.MCC':>9}{'off.DICE':>10}{'off.HD95':>10}{'off.복합':>10}"
          f"{'cov.MCC':>10}{'cov.DICE':>10}{'cov.복합':>10}{'cov.top1':>10}{'커버':>7}")
    for tag,o,c,t1,ct1,cov in sorted(rows,key=lambda r:-r[1]['MCC']):
        cs=f"{c['MCC']:>10.4f}{c['DICE']:>10.4f}{comp(c):>10.4f}" if c else f"{'-':>10}{'-':>10}{'-':>10}"
        print(f"{tag:<18}{o['MCC']:>9.4f}{o['DICE']:>10.4f}{o['HD95']:>10.4f}{comp(o):>10.4f}{cs}"
              f"{(ct1 or 0):>10.3f}{(cov or 0):>7.3f}")
PYEOF
log "=== 완료 ==="
