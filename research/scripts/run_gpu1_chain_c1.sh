#!/usr/bin/env bash
# GPU1 체인 (사용자 지시 2026-08-13): "A계열·V계열 최고 성능모델로 C1 다시 돌려보자"
#
# 문제: 기존 A5-2는 Dataset720의 nnU-Net 자동 5-fold(417케이스 무작위 333/84)로 학습돼
#       **공식 test 83 중 59개를 이미 학습에 봤다** -> 공식 평가 불가 (해당 실험은 _TESTLEAK로 격리).
# 해결: make_splits_417.py로 splits_final.json 재생성 (fold0 = 공식 split train292/val42,
#       test 83은 전 fold 배제) 후, A5-2와 같은 레시피를 fold0으로 재학습.
# V계열(V4-2)은 Dataset800이 애초에 공식 split이라 test 유출 없음 -> 재학습 불필요, 추론만.
#
# 순서: 1) A5-2 fold0 재학습  2) 동맥류 추론(val+test)  3) 혈관 추론(val+test, V4-2)
#       4) 혈관 후처리(V5 레시피)  5) C1/C2 평가(실제 예측 기반)
# 최종 성적은 test 83 기준으로 본다(val 42는 fold0 검증셋이라 참고용).
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ANALYSIS="$TOPANEU_ROOT/code/sblee/nnunet/analysis"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
ENVBIN2="$HOME/miniconda3/envs/sbaneu2/bin"
LOG="$TOPANEU_ROOT/experiments/gpu1_chain_c1.log"
E="$TOPANEU_ROOT/experiments"
WORK="$E/_c1_realpred"
EXP_A="A5-2_resencl_plainz_topk_417_f0"

export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
log() { echo "[gpu1-c1 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$WORK"

# ---------------------------------------------------------------- 입력 이미지 모으기
prep_inputs() {  # $1=출력디렉토리, $2=split키(val|test)
  "$ENVBIN/python" - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path
out, key = Path(sys.argv[1]), sys.argv[2]
root = Path(os.environ["TOPANEU_ROOT"])
ids = json.load(open(root/"dataset"/"TopAneu"/"dataset_split.json"))["splits"][key]
img = root/"nnunet"/"nnUNet_raw"/"Dataset720_TopAneuBinary417"/"imagesTr"
out.mkdir(parents=True, exist_ok=True)
n = 0
for cid in ids:
    s = img/f"{cid}_0000.nii.gz"
    d = out/f"{cid}_0000.nii.gz"
    if s.exists() and not d.exists():
        os.symlink(s, d); n += 1
print(f"[prep] {key}: {n}/{len(ids)} 링크 -> {out}")
PY
}

# ---------------------------------------------------------------- STEP1: 누출없는 동맥류 모델
log "STEP1: $EXP_A 학습 (A5-2와 동일 레시피, 공식 split fold0)"
GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 "$EXP_A" \
  -p nnUNetResEncUNetLPlans -tr nnUNetTrainerTverskyTopkCE >> "$LOG" 2>&1
S1=$?
DA=$(grep -h "Mean Validation Dice" "$E/$EXP_A"/results/*/*/fold_0/training_log_*.txt 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP1 종료 status=$S1 Dice=${DA:-N/A}  (참고: A5-2 fold0=0.5273, 단 split이 달라 직접비교 아님)"
[ $S1 -eq 0 ] || { log "ABORT: 동맥류 학습 실패"; exit 1; }
"$ENVBIN/python" "$SCRIPTS/make_summary.py" "$E/$EXP_A" >> "$LOG" 2>&1

# ---------------------------------------------------------------- STEP2/3: 추론
for SP in val test; do
  prep_inputs "$WORK/in_$SP" "$SP" >> "$LOG" 2>&1

  log "STEP2($SP): 동맥류 추론 ($EXP_A, fold0)"
  nnUNet_results="$E/$EXP_A/results" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
    -i "$WORK/in_$SP" -o "$WORK/aneu_$SP" -d 720 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTverskyTopkCE -p nnUNetResEncUNetLPlans -chk checkpoint_best.pth \
    --disable_tta -npp 2 -nps 2 >> "$LOG" 2>&1
  log "STEP2($SP) 종료 status=$?"

  log "STEP3($SP): 혈관 추론 (V4-2 classweighted v2, fold0)"
  nnUNet_results="$E/V4-2_vessel_classweighted_417_500ep/results" CUDA_VISIBLE_DEVICES=1 "$ENVBIN2/nnUNetv2_predict" \
    -i "$WORK/in_$SP" -o "$WORK/ves_$SP" -d 800 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep -p nnUNetResEncUNetMPlans \
    -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 >> "$LOG" 2>&1
  log "STEP3($SP) 종료 status=$?"

  log "STEP4($SP): 혈관 후처리 (V5 레시피)"
  "$ENVBIN/python" "$SCRIPTS/postprocess_vessel.py" apply "$WORK/ves_$SP" "$WORK/vespp_$SP" >> "$LOG" 2>&1
done

# ---------------------------------------------------------------- STEP5: C1/C2 평가
cd "$SCRIPTS"
for SP in val test; do
  for VES in "vespp_$SP:pp" "ves_$SP:raw"; do
    D="${VES%%:*}"; T="${VES##*:}"
    log "STEP5: C1(signature lookup) $SP / vessel=$T"
    "$ENVBIN/python" d900_infer_eval.py --split "$SP" \
      --aneurysm-pred-dir "$WORK/aneu_$SP" --vessel-pred-dir "$WORK/$D" \
      --lookup "$ANALYSIS/d9xx_lookup_V2_nearest_r2.json" \
      --tag "realpred_${T}" >> "$LOG" 2>&1
    log "STEP5: C2(weighted-kNN) $SP / vessel=$T"
    "$ENVBIN/python" d910_infer_eval.py --split "$SP" \
      --aneurysm-pred-dir "$WORK/aneu_$SP" --vessel-pred-dir "$WORK/$D" \
      --index "$ANALYSIS/d910_index_k5_p1.npz" \
      --tag "realpred_${T}" >> "$LOG" 2>&1
  done
done
log "GPU1 체인 완료 — 결과: analysis/d9xx_eval_*_realpred_*.json, d910_eval_*_realpred_*.json"
