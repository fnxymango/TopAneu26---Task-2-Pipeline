#!/usr/bin/env bash
# GPU0 체인 — 공식 split 재생성 후 재실행 (2026-08-13).
# 구 실험들은 Dataset720의 nnU-Net 자동 5-fold로 test 83 중 59개가 유출돼 있었음(_TESTLEAK로 격리).
#   1) A6-2 (ResEncL + TopAneuAdaptiveNorm + TverskyTopkCE), 공식 split fold0
#      -> GPU1에서 도는 A5-2(plain-z + topk, 같은 split)와 짝지어 'adaptive norm 효과'를 답함
#   2) A4(lesionscale) 전체볼륨 평가 — crop 내부평가(0.5131)와 잣대가 다른 문제 해소
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/gpu0_chain_a6.log"
E="$TOPANEU_ROOT/experiments"
EXP="A6-2_resencl_adaptivenorm_topk_417_f0"
export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
log() { echo "[gpu0-chain $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "STEP1: $EXP 학습 (공식 split fold0)"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 "$EXP" \
  -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE >> "$LOG" 2>&1
S1=$?
D=$(grep -h "Mean Validation Dice" "$E/$EXP"/results/*/*/fold_0/training_log_*.txt 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP1 종료 status=$S1 adaptive+topk Dice=${D:-N/A}  (짝: GPU1의 A5-2 plain-z+topk)"
"$ENVBIN/python" "$SCRIPTS/make_summary.py" "$E/$EXP" >> "$LOG" 2>&1
"$ENVBIN/python" "$SCRIPTS/make_leaderboard.py" >> "$LOG" 2>&1

log "STEP2: A4(lesionscale) 전체볼륨 평가"
bash "$SCRIPTS/run_a4_lesionscale_fullvolume.sh" >> "$LOG" 2>&1
log "GPU0 체인 완료"
