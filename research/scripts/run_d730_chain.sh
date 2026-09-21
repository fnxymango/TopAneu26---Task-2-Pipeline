#!/usr/bin/env bash
# D730(혈관ROI마스킹crop) 전체 체인: 빌드완료 대기 -> 전처리 -> fold0(250ep, D720과 공정비교)
# -> 1000ep 재학습 -> 5-fold 앙상블(1000ep, fold1~4). GPU0에서 순차 실행, 중간에 사람 개입 불필요.
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
PY="$ENVBIN/python"
LOG="$TOPANEU_ROOT/experiments/d730_chain.log"
mkdir -p "$TOPANEU_ROOT/experiments"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export TOPANEU_DATA="${TOPANEU_DATA:-$TOPANEU_ROOT/dataset/TopAneu}"

log() { echo "[d730 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

if [ "${SKIP_BUILD_WAIT:-0}" != "1" ]; then
  BUILD_PID=2831723
  log "빌드(PID $BUILD_PID) 종료 대기"
  while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 30; done
fi
[ -d "$nnUNet_raw/Dataset730_TopAneuBinaryVesselCrop417/imagesTr" ] || die "raw 빌드 실패"
log "빌드 확인 완료 (재개: dataset.json 버그 수정 후 전처리부터 재시작)"

# ---------------------------------------------------------------- 전처리
log "fingerprint+plan+preprocess 시작"
"$ENVBIN/nnUNetv2_plan_and_preprocess" -d 730 -c 3d_fullres --verify_dataset_integrity -np 4 \
  >> "$LOG" 2>&1 || die "전처리 실패"
log "전처리 완료"

log "splits 생성 (5-fold)"
"$PY" "$TOPANEU_ROOT/rebuild_417/make_splits_417.py" 730 >> "$LOG" 2>&1 || die "splits 실패"

# ---------------------------------------------------------------- STEP8: fold0 250ep (D720과 공정비교)
log "STEP8 시작: D730 fold0 250ep (base 레시피)"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 730 3d_fullres 0 D730_binary_vesselcrop_417 \
  -tr nnUNetTrainerTverskyTopkCE >> "$LOG" 2>&1
STEP8_STATUS=$?
D730_F0_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D730_binary_vesselcrop_417/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP8 종료 status=$STEP8_STATUS D730_fold0_Dice=${D730_F0_DICE:-N/A} (D720_fold0_기준=0.3305)"

# ---------------------------------------------------------------- STEP9: 1000ep
log "STEP9 시작: D730 fold0 1000ep"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 730 3d_fullres 0 D730_binary_vesselcrop_417_1000ep \
  -tr nnUNetTrainerTverskyTopkCE_1000ep >> "$LOG" 2>&1
STEP9_STATUS=$?
D730_1000EP_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D730_binary_vesselcrop_417_1000ep/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP9 종료 status=$STEP9_STATUS D730_1000ep_Dice=${D730_1000EP_DICE:-N/A}"

# ---------------------------------------------------------------- STEP10: 5-fold 앙상블 (fold1~4, 1000ep)
log "STEP10 시작: 5-fold 앙상블 (fold1~4, 1000ep)"
for F in 1 2 3 4; do
  log "  fold$F 시작"
  GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
    bash "$SCRIPTS/run_experiment.sh" 730 3d_fullres "$F" "D730_binary_vesselcrop_417_1000ep_f${F}" \
    -tr nnUNetTrainerTverskyTopkCE_1000ep >> "$LOG" 2>&1
  FS=$?
  FD=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D730_binary_vesselcrop_417_1000ep_f${F}/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  log "  fold$F 종료 status=$FS Dice=${FD:-N/A}"
done

log "D730 체인 전체 완료 (STEP8/9/10)"
