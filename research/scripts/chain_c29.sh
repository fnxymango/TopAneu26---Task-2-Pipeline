#!/usr/bin/env bash
# C29 — β 미세조정 + 분류 천장 측정, 그리고 GPU 로 혈관 fold1 학습 (2026-08-17).
#
# 지금까지: β=1.0 -> 0.5 로 test official MCC 0.2175 -> 0.2238, covered(÷36) 0.3141 -> 0.3233.
# 검출 동작점은 이미 최적(임계·필터 12종 전부 기본형 이하). 크롭 노선은 기각.
# 남은 것은 (a) β 를 0.5 주변에서 더 정밀하게, (b) 분류 천장이 어디인지, (c) 혈관마스크 품질.
#
# (c) 가 유일하게 남은 구조적 레버다 — 우리 피처 112차원이 전부 혈관마스크에서 나온다.
# 혈관은 fold0 단일 모델뿐이라 5-fold 앙상블한 검출기와 대칭이 안 맞는다. fold1 을 학습해 둔다.
# 500ep 에 ~30시간 걸리므로 CPU 작업과 완전히 병렬로 돌린다.
#
# **선택은 val, 보고는 test.** test 로 하이퍼파라미터를 고르면 그 test 숫자가 held-out 이 아니게 된다.
# 보고 표는 test 만, official_div52 와 covered_gt(=adjusted_div_present, ÷36) 를 함께 낸다(PROJECT_RULES.md 6-1).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN="$HOME/miniconda3/envs/sbaneu2/bin"
FEAT="$A/c10_feat_train.json"
VEXP="V4-3_vessel_classweighted_417_500ep_f1"
log(){ echo "[c29 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── GPU: 혈관 fold1 학습을 먼저 띄운다 (CPU 작업과 병렬) ──────────────
VLOG="$E/c29_vessel_f1.log"
NOW=$(date +%s); NEWEST=0
for f in "$VLOG" "$E/$VEXP/train.log"; do
  [ -e "$f" ] && { m=$(stat -c %Y "$f" 2>/dev/null || echo 0); [ "$m" -gt "$NEWEST" ] && NEWEST=$m; }
done
if [ -f "$E/$VEXP/results"/*/*/fold_1/checkpoint_final.pth ] 2>/dev/null; then
  log "혈관 fold1 이미 완료"
elif [ "$NEWEST" -gt 0 ] && [ $(( NOW - NEWEST )) -lt 3600 ]; then
  log "혈관 fold1 학습이 이미 진행 중 (갱신 $(( NOW - NEWEST ))s 전) — 재시작 안 함"
else
  log "GPU: 혈관 fold1 학습 시작 (~30시간, 백그라운드)"
  GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" nohup bash "$S/run_experiment.sh" \
    800 3d_fullres 1 "$VEXP" \
    -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep \
    > "$VLOG" 2>&1 &
  disown
fi

ev(){ # $1=split $2=tag $3=beta $4=ceil(1이면 GT천장)
  local sp=$1 tag=$2 bt=$3 ceil=${4:-0} VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  local ADARG="--aneurysm-pred-dir $AD"
  [ "$ceil" = 1 ] && ADARG=""            # 검출 없이 GT 병변으로 -> 분류 천장
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag (β=$bt${ceil:+ ceil=$ceil})"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" $ADARG \
    --model rf --use-pos --beta "$bt" --tag "$tag" 2>&1 \
    | grep -E '"MCC"|"top1_accuracy"' | head -2 || log "    실패"
}

# ── STEP1: val 에서 β 미세 스윕 (선택용) ──────────────────────────────
log "=== STEP1: val β 미세 스윕 (0.35~0.65) ==="
for B in 0.35 0.4 0.45 0.55 0.6 0.65; do ev val "c29_b$B" "$B"; done

BB=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
best=("0.5",-1)
pats=[("c5_eval_val_c29_b%s.json",None)]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c29_b*.json")) + \
         [os.path.join(sys.argv[1],"c5_eval_val_c26_b0.5.json")]:
    if not os.path.exists(f): continue
    m=re.search(r"_b([0-9.]+)\.json$",f)
    if not m: continue
    o=json.load(open(f)).get("official_div52") or {}
    if o.get("MCC",-1)>best[1]: best=(m.group(1),o["MCC"])
print(best[0])
PYEOF
)
log "=== val 최적 β = $BB ==="

# ── STEP2: test 평가 + 분류 천장 ──────────────────────────────────────
log "=== STEP2: test 평가 (β=$BB) 및 분류 천장 ==="
ev test "c29_b$BB" "$BB"
ev test "c29_ceil_b$BB" "$BB" 1
ev test "c29_ceil_b0.5" 0.5 1

# ── STEP3: 요약 — test 만, 두 분모 ────────────────────────────────────
log "=== STEP3: test 요약 (official_div52 / covered_gt ÷36) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c2[6789]*.json"))):
    d=json.load(open(f))
    if d.get("eval_mode")=="covered_gt": continue
    o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    tag=os.path.basename(f)[len("c5_eval_test_"):-5]
    rows.append((tag,o,c,d.get("n_present_classes_in_split"),d.get("aneurysm_pred_dir","")))
if rows:
    print(f"\n===== test  (covered_gt 분모 = {rows[0][3]}클래스) =====")
    print(f"{'설정':<18}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
          f"{'cov.Dice':>9}{'cov.VolSim':>11}{'cov.HD95':>9}{'cov.복합':>9}")
    for tag,o,c,n,ad in sorted(rows,key=lambda r:-r[2]['MCC']):
        mark=" (천장)" if "GT" in str(ad) else ""
        print(f"{tag+mark:<18}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
              f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['VOLSIM']:>11.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}")
PYEOF

# ── STEP4: 혈관 fold1 완료 대기 ───────────────────────────────────────
# 감시자는 로그가 1200초 안 변하면 정체로 보고 죽인다. 학습은 30시간이라
# **10분마다 하트비트를 찍어** 정체로 오인되지 않게 한다. 재기동돼도 모든 단계가 멱등이라 안전.
VRES="$E/$VEXP/results/Dataset800_TopAneuVessel417/nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres"
V0RES="$E/V4-2_vessel_classweighted_417_500ep/results/Dataset800_TopAneuVessel417/nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres"
log "=== STEP4: 혈관 fold1 학습 완료 대기 (~30시간, 10분마다 하트비트) ==="
for i in $(seq 1 300); do
  [ -f "$VRES/fold_1/checkpoint_final.pth" ] && { log "  혈관 fold1 완료"; break; }
  ep=$(grep -c "^Epoch" "$E/$VEXP/train.log" 2>/dev/null || echo 0)
  log "  (대기 $((i*10))분) fold1 epoch~${ep}/500"
  sleep 600
done

# ── STEP5: 혈관 2-fold 앙상블 -> 후처리 -> 분기점 -> test 재평가 ──────
if [ -f "$VRES/fold_1/checkpoint_final.pth" ] && [ -d "$V0RES/fold_0" ]; then
  log "=== STEP5: 혈관 2-fold 앙상블 ==="
  ln -sfn "$VRES/fold_1" "$V0RES/fold_1"       # 두 fold 를 한 results 트리에 모아야 -f 0 1 이 된다
  NN="$HOME/miniconda3/envs/sblee_topaneu/bin/nnUNetv2_predict"
  for SP in val test; do
    if [ "$(ls "$P/ves2_$SP"/*.nii.gz 2>/dev/null | wc -l)" -lt 40 ]; then
      log "  혈관 2-fold 추론 ($SP)"
      nnUNet_results="$E/V4-2_vessel_classweighted_417_500ep/results" CUDA_VISIBLE_DEVICES=0 \
        "$NN" -i "$P/in_$SP" -o "$P/ves2_$SP" -d 800 -c 3d_fullres -f 0 1 \
        -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep \
        -chk checkpoint_best.pth --disable_tta --continue_prediction -npp 2 -nps 2 \
        > "$E/c29_ves2_$SP.log" 2>&1 || log "    추론 실패"
    fi
    if [ "$(ls "$P/ves2pp_$SP"/*.nii.gz 2>/dev/null | wc -l)" -lt 40 ]; then
      log "  혈관 후처리 ($SP)"
      $PY -u postprocess_vessel.py apply "$P/ves2_$SP" "$P/ves2pp_$SP" 2>&1 | tail -3 || log "    후처리 실패"
    fi
    if [ "$(ls "$BP/ves2pp_$SP"/*.json 2>/dev/null | wc -l)" -lt 40 ]; then
      log "  분기점 그래프 ($SP)"
      $PY -u c4_branchpoint_graph.py --split "$SP" --vessel-dir "$P/ves2pp_$SP" \
         --out "$BP/ves2pp_$SP" 2>&1 | tail -3 || log "    그래프 실패"
    fi
  done
  log "=== STEP5: 새 혈관으로 test 재평가 ==="
  BB2=${BB:-0.5}
  if [ ! -f "$A/c5_eval_test_c29_ves2_b$BB2.json" ]; then
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split test \
      --vessel-dir "$P/ves2pp_test" --bp-dir "$BP/ves2pp_test" \
      --aneurysm-pred-dir "$P/aneu_test_probavgf" \
      --model rf --use-pos --beta "$BB2" --tag "c29_ves2_b$BB2" 2>&1 \
      | grep -E '"MCC"' | head -1 || log "  평가 실패"
  fi
  log "=== STEP6: 최종 test 표 ==="
  $PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c2[6789]*.json"))):
    d=json.load(open(f))
    if d.get("eval_mode")=="covered_gt": continue
    o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5],o,c))
if rows:
    print(f"\n{'설정':<20}| {'off.MCC':>8}{'off.복합':>9} | {'cov.MCC':>8}{'cov.복합':>9}")
    for tag,o,c in sorted(rows,key=lambda r:-r[2]['MCC']):
        print(f"{tag:<20}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['MCC']:>8.4f}{comp(c):>9.4f}")
PYEOF
else
  log "혈관 fold1 미완 — STEP5 생략"
fi
log "=== 완료 ==="
