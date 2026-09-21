#!/usr/bin/env bash
# MCC 상승을 목표로 한 후속 실험 (2026-08-16). **val 로만 평가**, test 는 승자 하나만 나중에.
#   C17  학습된 FP 기각기 적용        — FP 제거 → precision ↑ → MCC ↑ (직접)
#   C20  확률맵 임계 스윕             — 검출 동작점을 MCC로 최적화 (직접)
#   C19  A4 경계 병합                 — DICE/HD95용. MCC를 깎지 않는지 확인용
# 대기 조건은 절대 pgrep 으로 걸지 않는다 (앞서 두 번 정체시킨 원인). 산출물 개수로 판정.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
NN="$HOME/miniconda3/envs/sblee_topaneu/bin/nnUNetv2_predict"
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
FEAT="$A/c10_feat_train.json"; BEST="--model rf --beta 1.0 --use-pos"
M5="$E/_c16_a62_5fold/results"
log(){ echo "[mcc $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── C20-a: 확률 저장 재추론 (val) ──────────────────────────────────────
if [ "$(ls "$P"/aneu_val_prob/*.npz 2>/dev/null | wc -l)" -lt 42 ]; then
  log "C20: val 5-fold 확률맵 재추론 (--save_probabilities)"
  nnUNet_results="$M5" CUDA_VISIBLE_DEVICES=0 "$NN" -i "$P/in_val" -o "$P/aneu_val_prob" \
    -d 720 -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerTverskyTopkCE \
    -p nnUNetResEncUNetLPlansAdaptive -chk checkpoint_best.pth --disable_tta \
    --save_probabilities -npp 2 -nps 2 > "$E/c20_pred_val.log" 2>&1 || log "  재추론 실패"
fi
log "C20 확률맵 $(ls "$P"/aneu_val_prob/*.npz 2>/dev/null | wc -l)/42"

# ── C17: FP 기각기 적용 (val) ─────────────────────────────────────────
for TH in 0.1 0.2 0.3; do
  OUT="$P/aneu_val_fpr${TH}"
  log "C17: FP 기각기 적용 (임계 $TH)"
  $PY -u c17_fp_reject.py apply --cand "$A/c17_candidates.json" \
     --aneu-dir "$P/aneu_val_probavgf" --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" \
     --split val --out "$OUT" --thresh "$TH" 2>&1 | tail -2 || log "  실패"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split val \
     --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --aneurysm-pred-dir "$OUT" \
     $BEST --tag "fpr${TH}" 2>&1 | grep -E '"MCC"|"DICE"' | head -2 || log "  평가 실패"
done

# ── C20-b: 임계 스윕 ──────────────────────────────────────────────────
if [ "$(ls "$P"/aneu_val_prob/*.npz 2>/dev/null | wc -l)" -ge 42 ]; then
  log "C20: 임계 스윕"
  $PY -u c20_prob_threshold.py --prob-dir "$P/aneu_val_prob" --vessel-dir "$P/vespp_val" \
     --split val --ref-dir "$P/aneu_val_probavg" --out-root "$P/_c20_val" \
     --thresholds 0.3,0.4,0.5,0.6,0.7 2>&1 | tail -12 || log "  스윕 실패"
  for T in 0.3 0.4 0.6 0.7; do
    [ -d "$P/_c20_val/th$T" ] || continue
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split val \
       --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --aneurysm-pred-dir "$P/_c20_val/th$T" \
       $BEST --tag "c20_th$T" 2>&1 | grep -E '"MCC"' | head -1 || log "  th$T 평가 실패"
  done
fi

# ── C19: A4 경계 병합 (MCC 영향 확인용) ───────────────────────────────
log "C19: A4 경계 병합 (val)"
$PY -u c19_merge_a4.py --det-dir "$P/aneu_val_probavgf" --a4-dir "$P/a4_val" \
   --split val --out "$P/aneu_val_a4refine" 2>&1 | tail -3 || log "  병합 실패"
$PY -u c5_location_v2.py eval --train-feat "$FEAT" --split val \
   --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --aneurysm-pred-dir "$P/aneu_val_a4refine" \
   $BEST --tag "a4refine" 2>&1 | grep -E '"MCC"|"DICE"|"HD95"' | head -3 || log "  평가 실패"

# ── 요약 (MCC 내림차순) ───────────────────────────────────────────────
log "=== val 요약 (MCC 기준) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_*.json")):
    if "SMOKE" in f: continue
    d=json.load(open(f)); o=d.get("official_div52")
    if not o: continue
    n=os.path.basename(f).replace("c5_eval_val_","").replace(".json","")
    rows.append((n,o["MCC"],o["DICE"],o["HD95"],(sum(o[k] for k in K)+1-o["HD95"])/6))
rows.sort(key=lambda r:-r[1])
print(f"{'설정':<26}{'MCC':>9}{'DICE':>9}{'HD95':>9}{'복합':>9}")
for n,m,dd,h,c in rows[:12]:
    print(f"{n:<26}{m:>9.4f}{dd:>9.4f}{h:>9.4f}{c:>9.4f}")
PYEOF
log "=== 완료 ==="
