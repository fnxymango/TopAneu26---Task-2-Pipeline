#!/usr/bin/env bash
# D720 ResEncL 재현 체인 STEP5/6 재개 (2026-08-12).
# 원래 run_d720_resencl_repro.sh는 STEP4(adaptive-norm 전처리) 도중 디스크 100% ->
# BrokenPipe로 죽었음. 이후 Dataset800 preprocessed를 HDD(/mnt/hdd/sblee)로 옮겨
# SSD 105GB 확보 + adaptive 전처리 417케이스 완료 확인됨 -> STEP5/6만 이어서 실행.
# 원본 체인은 GPU0 순차였지만 지금 GPU 둘 다 유휴라 병렬로 돌려 wall-clock 절반.
#   STEP5 = adaptive-norm + plain trainer  -> GPU0
#   STEP6 = adaptive-norm + topk  trainer  -> GPU1
# 비교기준: plainZ+plain=0.5242, plainZ+topk=0.5273 (STEP2/3 완료분)
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d720_step56_parallel.log"
mkdir -p "$TOPANEU_ROOT/experiments"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"

log() { echo "[d720-step56 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "STEP5(GPU0, adaptive+plain) / STEP6(GPU1, adaptive+topk) 병렬 시작"

GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_adaptivenorm_plain_417_f0 \
  -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainer_250epochs >> "$LOG" 2>&1 &
PID5=$!
log "  STEP5 PID=$PID5"

GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_adaptivenorm_topk_417_f0 \
  -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE >> "$LOG" 2>&1 &
PID6=$!
log "  STEP6 PID=$PID6"

wait $PID5; S5=$?
wait $PID6; S6=$?

D5=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_adaptivenorm_plain_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
D6=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_adaptivenorm_topk_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')

log "STEP5 종료 status=$S5 adaptive_plain_Dice=${D5:-N/A}"
log "STEP6 종료 status=$S6 adaptive_topk_Dice=${D6:-N/A}"
log "4종 최종비교: plainZ+plain=0.5242  plainZ+topk=0.5273  adaptive+plain=${D5:-N/A}  adaptive+topk=${D6:-N/A}  (외부claim=0.82, D720 기존최고=0.4157)"
