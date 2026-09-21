#!/usr/bin/env bash
# D740 체인: plan_and_preprocess -> fold0 500ep 학습 (D730과 동일 레시피, D720/D730 스크리닝 비교용)
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d740_chain.log"
mkdir -p "$TOPANEU_ROOT/experiments"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export TOPANEU_DATA="${TOPANEU_DATA:-$TOPANEU_ROOT/dataset/TopAneu}"

log() { echo "[d740 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

log "STEP1 시작: plan_and_preprocess Dataset740"
"$ENVBIN/nnUNetv2_plan_and_preprocess" -d 740 -c 3d_fullres --verify_dataset_integrity -np 4 >> "$LOG" 2>&1
STEP1_STATUS=$?
[ $STEP1_STATUS -eq 0 ] || die "STEP1 실패 status=$STEP1_STATUS"
log "STEP1 완료"

log "STEP2 시작: D740 fold0 500ep 학습 (GPU0)"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 740 3d_fullres 0 D740_binary_lesionscale_417_500ep \
  -tr nnUNetTrainerTverskyTopkCE_500ep >> "$LOG" 2>&1
STEP2_STATUS=$?
D740_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D740_binary_lesionscale_417_500ep/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP2 종료 status=$STEP2_STATUS D740_fold0_500ep_Dice=${D740_DICE:-N/A} (D720_fold0=0.3305, D730_fold0=0.3218)"
