#!/usr/bin/env bash
# D800(vessel, ResEnc-M) 발전판: 250ep로 이미 Dice 0.703 확보 -> 전처리 재생성(디스크정리로 캐시
# 삭제됐었음) -> fold0 1000ep 재학습 -> 5-fold 앙상블(fold1~4, 1000ep). GPU1에서 순차 실행.
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sbaneu2/bin"
PY="$ENVBIN/python"
LOG="$TOPANEU_ROOT/experiments/d800_1000ep_chain.log"
mkdir -p "$TOPANEU_ROOT/experiments"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export TOPANEU_DATA="${TOPANEU_DATA:-$TOPANEU_ROOT/dataset/TopAneu}"

log() { echo "[d800 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

# ---------------------------------------------------------------- 전처리 (디스크정리로 캐시 삭제됐었음, 재생성)
if [ ! -d "$nnUNet_preprocessed/Dataset800_TopAneuVessel417/nnUNetResEncUNetMPlans_3d_fullres" ]; then
  log "fingerprint+plan(ResEnc-M)+preprocess 시작"
  "$ENVBIN/nnUNetv2_plan_and_preprocess" -d 800 -c 3d_fullres -pl nnUNetPlannerResEncM \
    --verify_dataset_integrity -np 4 >> "$LOG" 2>&1 || die "전처리 실패"
  log "전처리 완료"
else
  log "전처리 이미 존재, 스킵"
fi

log "splits 생성 (5-fold)"
"$PY" "$TOPANEU_ROOT/rebuild_417/make_splits_417.py" 800 >> "$LOG" 2>&1 || die "splits 실패"

# ---------------------------------------------------------------- STEPA: fold0 1000ep (D800 250ep=0.7027과 비교)
log "STEPA 시작: D800 fold0 1000ep"
GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 800 3d_fullres 0 D800_vessel_skelrec_resencm_417_1000ep \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroring_1000ep >> "$LOG" 2>&1
STEPA_STATUS=$?
D800_1000EP_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D800_vessel_skelrec_resencm_417_1000ep/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEPA 종료 status=$STEPA_STATUS D800_1000ep_Dice=${D800_1000EP_DICE:-N/A} (D800_250ep_기준=0.7027)"

# ---------------------------------------------------------------- STEPB: 5-fold 앙상블 (fold1~4, 1000ep)
log "STEPB 시작: 5-fold 앙상블 (fold1~4, 1000ep)"
for F in 1 2 3 4; do
  log "  fold$F 시작"
  GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
    bash "$SCRIPTS/run_experiment.sh" 800 3d_fullres "$F" "D800_vessel_skelrec_resencm_417_1000ep_f${F}" \
    -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroring_1000ep >> "$LOG" 2>&1
  FS=$?
  FD=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D800_vessel_skelrec_resencm_417_1000ep_f${F}/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  log "  fold$F 종료 status=$FS Dice=${FD:-N/A}"
done

log "D800 1000ep 체인 전체 완료 (STEPA/B)"
