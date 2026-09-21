#!/usr/bin/env bash
# D730 체인 이어받기: 1000ep -> 500ep로 전환(사용자 지시, 2026-08-10).
# STEP8(fold0 250ep, 원래 wrapper가 이미 진행중)은 그대로 두고, 그 뒤 STEP9/10만
# 500ep로 바꿔서 새로 이어붙인다. 원래 run_d730_chain.sh 최상위 wrapper(PID로 감시)는
# kill했지만 STEP8 학습 프로세스 자체는 안 건드림(자식이라 안 죽음).
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

log() { echo "[d730-500ep $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

STEP8_PID="${STEP8_WAIT_PID:-2855708}"
if [ "${SKIP_STEP8_WAIT:-0}" != "1" ]; then
  log "STEP8(fold0 250ep, PID $STEP8_PID) 종료 대기 (1000ep->500ep 전환)"
  while kill -0 "$STEP8_PID" 2>/dev/null; do sleep 30; done
fi
D730_F0_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D730_binary_vesselcrop_417/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP8 확인 완료 D730_fold0_250ep_Dice=${D730_F0_DICE:-N/A} (D720_fold0_기준=0.3305)"

# ---------------------------------------------------------------- STEP9: fold0 500ep
log "STEP9 시작: D730 fold0 500ep"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
  bash "$SCRIPTS/run_experiment.sh" 730 3d_fullres 0 D730_binary_vesselcrop_417_500ep \
  -tr nnUNetTrainerTverskyTopkCE_500ep >> "$LOG" 2>&1
STEP9_STATUS=$?
D730_500EP_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D730_binary_vesselcrop_417_500ep/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP9 종료 status=$STEP9_STATUS D730_500ep_Dice=${D730_500EP_DICE:-N/A} (250ep=${D730_F0_DICE:-N/A})"

# ---------------------------------------------------------------- 스크리닝 단계까지만.
# D740(다른 후보) fold0도 봐야 어느 쪽을 5fold 앙상블(STEP10, 비용 5배)에 태울지 판단
# 가능하므로, 여기서 멈춘다. STEP10은 D720/D730/D740 fold0 비교 후 별도로 수동 트리거.
log "D730 fold0 스크리닝 완료 (STEP8/9). 5fold 앙상블(STEP10)은 D740 fold0와 비교 후 결정 -> 지금은 대기"
