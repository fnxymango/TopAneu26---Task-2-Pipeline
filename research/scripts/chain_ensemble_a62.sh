#!/usr/bin/env bash
# A6-2 5-fold 3/5 다수결 앙상블 파이프라인 (2026-08-15).
# run_a62_5fold.sh 가 fold1~4를 다 끝내면 자동으로 이어 돈다.
#
#   1) fold1~4 로 val/test 동맥류 추론 (fold0 예측은 이미 있음)
#   2) 5모델 다수결 -> 투표수 M in {2,3,4} 별 마스크 생성
#   3) c7 후처리(혈관거리 게이팅 + 성분크기) 적용
#   4) 검출 지표(민감도/FP) 로 **val에서 M 선택** -> 그 M으로 test 평가
#      (splits --train-only 라 5모델 전원이 val42를 학습에 안 썼으므로 val이 깨끗한 선택셋)
#   5) 최종 C5 위치분류 평가
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
BEST="--model rf --beta 1.0 --use-pos"
LOG="$E/ensemble_a62.log"
log(){ echo "[ens $(date -u +'%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ---- 0) 학습 완료 대기 ----
log "fold1~4 학습 완료 대기"
while :; do
  done_n=0
  for f in 1 2 3 4; do
    d="$E/A6-2_resencl_adaptivenorm_topk_417_f$f"
    [ -f "$d"/results/*/*/fold_$f/checkpoint_final.pth ] 2>/dev/null && done_n=$((done_n+1))
  done
  [ "$done_n" -ge 4 ] && break
  pgrep -f 'run_a62_5fol[d]\.sh' >/dev/null || { log "학습 프로세스 없음 (완료 $done_n/4) — 진행"; break; }
  sleep 300
done
log "학습 완료 확인 ($done_n/4)"

# ---- 1) fold1~4 추론 ----
for f in 1 2 3 4; do
  EXP="A6-2_resencl_adaptivenorm_topk_417_f$f"
  [ -d "$E/$EXP/results" ] || { log "fold$f 결과 없음 — 건너뜀"; continue; }
  for SP in val test; do
    OUT="$P/aneu_${SP}_a62f${f}"
    [ -d "$OUT" ] && [ "$(ls "$OUT" 2>/dev/null | wc -l)" -gt 0 ] && { log "fold$f $SP 이미 있음"; continue; }
    log "추론 fold$f / $SP"
    nnUNet_results="$E/$EXP/results" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_$SP" -o "$OUT" -d 720 -c 3d_fullres -f "$f" \
      -tr nnUNetTrainerTverskyTopkCE -p nnUNetResEncUNetLPlansAdaptive \
      -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 >> "$LOG" 2>&1
    log "  fold$f $SP 종료 status=$?"
  done
done

# ---- 2) 다수결 + 3) c7 후처리 + 4) 검출지표 ----
for SP in val test; do
  DIRS="$P/aneu_${SP}_a62"
  for f in 1 2 3 4; do
    [ -d "$P/aneu_${SP}_a62f${f}" ] && DIRS="$DIRS,$P/aneu_${SP}_a62f${f}"
  done
  n_models=$(echo "$DIRS" | tr ',' '\n' | wc -l)
  log "$SP: 앙상블 멤버 $n_models개"
  for M in 2 3 4; do
    [ "$M" -gt "$n_models" ] && continue
    OUT="$P/aneu_${SP}_vote${M}"
    [ -d "$OUT" ] || $PY -u c6_consensus.py --in-dirs "$DIRS" --min-votes "$M" \
        --mode lesion --out "$OUT" >> "$LOG" 2>&1
    # c7 필터 (A6-2 최적: min_vox 5, dist 3mm)
    VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
    $PY -u c7_detect_postproc.py --aneu-dir "$OUT" --vessel-dir "$P/$VES" --split "$SP" \
        --tag "vote${M}_${SP}" --save-best "$P/aneu_${SP}_vote${M}f" --force-cfg 5,3 >> "$LOG" 2>&1
    log "  $SP vote$M 완료"
  done
done

# ---- val에서 최적 투표수 선택 (병변 민감도 - FP/case 균형) ----
BESTM=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys
best=None
for f in glob.glob(os.path.join(sys.argv[1],"c7_detect_sweep_vote*_val.json")):
    d=json.load(open(f))
    g=[r for r in d["grid"] if r["min_vox"]==5 and abs(r["max_dist_mm"]-3.0)<1e-6]
    if not g: continue
    r=g[0]
    # 엔드투엔드에서 FP 1개 비용이 미검출 1개보다 컸으므로(C3/C5 관찰) FP에 가중
    score=r["lesion_sensitivity"]-0.004*r["fp_total"]
    m=os.path.basename(f).split("vote")[1].split("_")[0]
    if best is None or score>best[0]: best=(score,m,r)
print(best[1] if best else "3")
PYEOF
)
log "val 선택 투표수 = $BESTM"

# ---- 5) 최종 위치분류 평가 ----
for SP in val test; do
  VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
  VB=$([ "$SP" = val ] && echo val_pred || echo vespp_test)
  log "C5 평가 $SP / vote$BESTM"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$SP" \
    --vessel-dir "$P/$VES" --bp-dir "$BP/$VB" \
    --aneurysm-pred-dir "$P/aneu_${SP}_vote${BESTM}f" \
    $BEST --tag "ens_vote${BESTM}" >> "$LOG" 2>&1
  log "  $SP 종료 status=$?"
done

log "=== 앙상블 파이프라인 완료 ==="
$PY - "$A" <<'PYEOF' >> "$LOG"
import json,glob,os,sys
print(f"{'실험':<34}{'MCC(adj)':>10}{'MCC(공식)':>11}{'DICE(adj)':>11}")
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_*_ens_*.json"))
              + glob.glob(os.path.join(sys.argv[1],"c5_eval_*_best_e2e*.json"))):
    d=json.load(open(f)); a=d.get("adjusted_div_present") or {}; o=d.get("official_div52") or {}
    n=os.path.basename(f).replace("c5_eval_","").replace(".json","")
    print(f"{n:<34}{a.get('MCC',0):>10.4f}{o.get('MCC',0):>11.4f}{a.get('DICE',0):>11.4f}")
PYEOF
