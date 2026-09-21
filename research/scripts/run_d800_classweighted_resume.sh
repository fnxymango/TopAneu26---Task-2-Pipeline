#!/usr/bin/env bash
# GPU1을 다른 사용자가 써야 해서 epoch~55에서 kill(2026-08-11) -> checkpoint_latest.pth(epoch50)
# 에서 --c(continue)로 재개. 같은 실험명 재사용 -> nnU-Net이 자동으로 checkpoint_latest.pth 로드.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sbaneu2/bin"
LOG="$TOPANEU_ROOT/experiments/d800_classweighted_resume.log"
mkdir -p "$TOPANEU_ROOT/experiments"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
log() { echo "[d800-classweighted-resume $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "재개: D800 fold0 500ep class-weighted, checkpoint_latest(epoch50)에서 --c"
GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 800 3d_fullres 0 V4_vessel_classweighted_417_500ep \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeighted_500ep --c >> "$LOG" 2>&1
STATUS=$?
DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/V4_vessel_classweighted_417_500ep/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "종료 status=$STATUS Dice=${DICE:-N/A} (기존 D800 500ep 기준=0.7620)"
