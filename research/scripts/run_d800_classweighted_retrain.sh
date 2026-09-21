#!/usr/bin/env bash
# L-P3P4 dead-class 붕괴(epoch105) 진단 후 해결책 검증: CE loss에 class weight 부여
# (L-P3P4=5x, 데이터부족 3종=3x) 재학습. GPU1(STEPB 취소로 유휴) 사용.
# 목적: 1) 재현되는 붕괴인지(같은 지점에서 다시 죽는지) 2) weight로 복구되는지 확인.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sbaneu2/bin"
LOG="$TOPANEU_ROOT/experiments/d800_classweighted_retrain.log"
mkdir -p "$TOPANEU_ROOT/experiments"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
log() { echo "[d800-classweighted $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "시작: D800 fold0 500ep class-weighted(L-P3P4=5x, 3rd-A2/3rd-A3/R-AChA=3x) 재학습"
GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 800 3d_fullres 0 V4_vessel_classweighted_417_500ep \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeighted_500ep >> "$LOG" 2>&1
STATUS=$?
DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/V4_vessel_classweighted_417_500ep/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "종료 status=$STATUS Dice=${DICE:-N/A} (기존 D800 500ep 기준=0.7620, 목표: L-P3P4 dead class 복구)"
