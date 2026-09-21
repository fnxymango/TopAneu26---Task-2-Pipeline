#!/usr/bin/env bash
# D800 체인 이어받기: 1000ep -> 500ep로 전환(사용자 지시, 2026-08-10).
# 재전처리(원래 wrapper가 이미 진행중)는 그대로 두고, splits 생성 + STEPA/B만 500ep로
# 새로 이어붙인다. 원래 run_d800_1000ep_chain.sh 최상위 wrapper는 kill했지만 전처리
# 프로세스 자체는 안 건드림(자식이라 안 죽음).
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sbaneu2/bin"
PY="$ENVBIN/python"
LOG="$TOPANEU_ROOT/experiments/d800_500ep_chain.log"
mkdir -p "$TOPANEU_ROOT/experiments"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export TOPANEU_DATA="${TOPANEU_DATA:-$TOPANEU_ROOT/dataset/TopAneu}"

log() { echo "[d800-500ep $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

PREP_PID="${PREP_WAIT_PID:-2863941}"
if [ "${SKIP_PREP_WAIT:-0}" != "1" ]; then
  log "재전처리(PID $PREP_PID) 종료 대기 (1000ep->500ep 전환)"
  while kill -0 "$PREP_PID" 2>/dev/null; do sleep 15; done
fi
# 주의: ResEncM 플랜이어도 3d_fullres의 실제 data_identifier는 plans json에 따라
# "nnUNetPlans_3d_fullres"로 저장될 수 있음(nnU-Net이 내부적으로 plans json을 보고
# 알아서 찾으므로 학습엔 문제없음, 여기선 존재 확인만 케이스 파일 개수로 함).
N_PP=$(ls "$nnUNet_preprocessed/Dataset800_TopAneuVessel417"/*_3d_fullres/*.npz 2>/dev/null | wc -l)
[ "$N_PP" -ge 400 ] || die "전처리 실패/미완료 (전처리된 케이스 $N_PP개)"
log "전처리 확인 완료"

log "splits 생성 (5-fold)"
"$PY" "$TOPANEU_ROOT/rebuild_417/make_splits_417.py" 800 >> "$LOG" 2>&1 || die "splits 실패"

# ---------------------------------------------------------------- STEPA: fold0 500ep (D800 250ep=0.7027과 비교)
log "STEPA 시작: D800 fold0 500ep"
GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 800 3d_fullres 0 D800_vessel_skelrec_resencm_417_500ep \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroring_500ep >> "$LOG" 2>&1
STEPA_STATUS=$?
D800_500EP_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D800_vessel_skelrec_resencm_417_500ep/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEPA 종료 status=$STEPA_STATUS D800_500ep_Dice=${D800_500EP_DICE:-N/A} (D800_250ep_기준=0.7027)"

# ---------------------------------------------------------------- STEPB: 5-fold 앙상블 (fold1~4, 500ep)
log "STEPB 시작: 5-fold 앙상블 (fold1~4, 500ep)"
for F in 1 2 3 4; do
  log "  fold$F 시작"
  GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
    bash "$SCRIPTS/run_experiment.sh" 800 3d_fullres "$F" "D800_vessel_skelrec_resencm_417_500ep_f${F}" \
    -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroring_500ep >> "$LOG" 2>&1
  FS=$?
  FD=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D800_vessel_skelrec_resencm_417_500ep_f${F}/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  log "  fold$F 종료 status=$FS Dice=${FD:-N/A}"
done

log "D800 500ep 체인 전체 완료 (STEPA/B)"
