#!/usr/bin/env bash
# C32 + A7 — 축을 돌린다 (2026-08-17). **한 스크립트**로, GPU/CPU 병렬.
#
# 오늘 채택된 것: C26(β 이중보정 교정) + C31(β 를 저확신 병변에만).
#   test cov.MCC 0.3188 -> 0.3233 -> 0.3306.
# 분류 결정규칙 축은 여기까지가 한계로 보인다. 두 개의 새 축으로 돈다.
#
#   C32 (CPU)  분류기 학습셋에 공식 val 42 추가 (병변 268 -> 311).
#              PROJECT_RULES.md §1 상 val 은 학습 사용 가능(held-out 은 test 83 뿐).
#              n<=2 클래스가 18 -> 15 로 줄어 희소 클래스에 직접 들어간다.
#              하이퍼파라미터(β=0.5, τ=0.5)는 **이미 val 로 고른 값을 고정**해서 쓴다.
#              val 이 학습에 들어가면 더는 선택에 못 쓰므로, 새로 고르지 않는다.
#
#   A7  (GPU)  Tversky alpha 0.3->0.15, beta 0.7->0.85 (FN 벌점 강화) 로 검출기 재학습.
#              검출이 test 에서 cov.MCC 0.120 을 먹고 있고(천장 0.4429 vs e2e 0.3306),
#              C27/C28 에서 임계 5종 x 필터 3종을 다 재본 결과 후처리로는 못 줄인다.
#              평가는 Dice 가 아니라 **c7 필터 적용 후 병변단위 민감도 / FP** 로 한다 —
#              FP 가 늘 것은 확실하고, 질문은 "필터 뒤에도 민감도 순증이 남는가" 다.
#
# 대기는 산출물/파일로만 판정하고 pgrep 을 쓰지 않는다. 10분마다 하트비트를 찍어
# 감시자가 정체로 오인하지 않게 한다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
AEXP="A7_tversky_a15b85_417_f0"
FEAT="$A/c10_feat_train.json"; MERGED="$A/c32_feat_trainval.json"
BETA=0.5; TAU=0.5
log(){ echo "[c32 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── GPU: A7 검출기 학습 먼저 띄운다 ───────────────────────────────────
ALOG="$E/c32_a7_train.log"
NOW=$(date +%s); NEWEST=0
for f in "$ALOG" "$E/$AEXP/train.log"; do
  [ -e "$f" ] && { m=$(stat -c %Y "$f" 2>/dev/null || echo 0); [ "$m" -gt "$NEWEST" ] && NEWEST=$m; }
done
ARES="$E/$AEXP/results/Dataset720_TopAneuBinary417/nnUNetTrainerTverskyTopkCE_a15b85__nnUNetResEncUNetLPlansAdaptive__3d_fullres"
if [ -f "$ARES/fold_0/checkpoint_final.pth" ]; then
  log "A7 이미 완료"
elif [ "$NEWEST" -gt 0 ] && [ $(( NOW - NEWEST )) -lt 3600 ]; then
  log "A7 학습이 이미 진행 중 (갱신 $(( NOW - NEWEST ))s 전) — 재시작 안 함"
else
  log "GPU: A7 검출기 학습 시작 (Tversky 0.15/0.85, 백그라운드)"
  GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" nohup bash "$S/run_experiment.sh" \
    720 3d_fullres 0 "$AEXP" \
    -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE_a15b85 \
    > "$ALOG" 2>&1 &
  disown
fi

ev(){ # $1=split $2=tag $3=train-feat
  local sp=$1 tag=$2 tf=$3 VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag"
  $PY -u c5_location_v2.py eval --train-feat "$tf" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
    --model rf --use-pos --beta "$BETA" --conf-tau "$TAU" --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

# ── C32: 학습셋 병합 후 test 평가 ─────────────────────────────────────
log "=== C32: 학습셋에 val 42 추가 ==="
if [ ! -f "$MERGED" ]; then
  $PY - "$FEAT" "$A/c10_feat_val_refves.json" "$MERGED" <<'PYEOF'
import json,sys,collections
tr=json.load(open(sys.argv[1])); va=[r for r in json.load(open(sys.argv[2])) if r.get("gt_loc")]
need=("dist_mm","overlap","bp_mm","pos","gt_loc")
va=[r for r in va if all(k in r for k in need)]
out=tr+va
json.dump(out,open(sys.argv[3],"w"),ensure_ascii=False)
g=[r for r in out if r.get("gt_loc")]
c=collections.Counter(r["gt_loc"] for r in g)
print(f"[c32] 병합 {len(tr)} + {len(va)} = {len(g)} 병변 · 클래스 {len(c)} · n<=2 클래스 {sum(1 for v in c.values() if v<=2)}")
PYEOF
fi
ev test "c32_trainval" "$MERGED"
ev test "c32_trainonly" "$FEAT"      # 동일 설정에서 학습셋만 다른 대조군

log "=== C32 결과 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c3[12]*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c26_b0.5.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f))
    if d.get("eval_mode")=="covered_gt": continue
    o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5],o,c,d.get("n_lesions_predicted")))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<18}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.HD95':>9}{'cov.복합':>9}")
for tag,o,c,n in rows:
    print(f"{tag:<18}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}")
PYEOF

# ── A7: 학습 완료 대기 → 검출 성능 비교 ───────────────────────────────
log "=== A7: 학습 완료 대기 (10분마다 하트비트) ==="
for i in $(seq 1 300); do
  [ -f "$ARES/fold_0/checkpoint_final.pth" ] && { log "  A7 학습 완료"; break; }
  TL=$(ls -t "$ARES"/fold_0/training_log_*.txt 2>/dev/null | head -1)
  ep=$([ -n "$TL" ] && grep -c "Epoch time" "$TL" 2>/dev/null || echo 0)
  log "  (대기 $((i*10))분) A7 epoch~${ep}"
  sleep 600
done

if [ -f "$ARES/fold_0/checkpoint_final.pth" ]; then
  log "=== A7: val 추론 + 검출 성능 비교 ==="
  if [ "$(ls "$P/aneu_val_a7"/*.nii.gz 2>/dev/null | wc -l)" -lt 40 ]; then
    nnUNet_results="$E/$AEXP/results" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val" -o "$P/aneu_val_a7" -d 720 -c 3d_fullres -f 0 \
      -tr nnUNetTrainerTverskyTopkCE_a15b85 -p nnUNetResEncUNetLPlansAdaptive \
      -chk checkpoint_best.pth --disable_tta --continue_prediction -npp 2 -nps 2 \
      > "$E/c32_a7_pred_val.log" 2>&1 || log "  추론 실패"
  fi
  log "  c7 필터 적용 후 민감도/FP (A6-2 5-fold 대비)"
  $PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_val_a7" --vessel-dir "$P/vespp_val" \
     --split val 2>&1 | tail -12 || log "  후처리 평가 실패"
else
  log "A7 미완 — 평가 생략"
fi
log "=== 완료 ==="
