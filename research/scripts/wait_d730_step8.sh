#!/usr/bin/env bash
# 사용자 지시(2026-08-10): D7xx 스크리닝은 250ep로만 apples-to-apples 비교, 500ep/5fold는
# 나중에(효과 확인되면). STEP8(fold0 250ep)만 기다렸다가 D720(0.3305) 대비 결과를 로그.
# 그 다음 단계(D740 fold0, 250ep)는 별도로 새로 빌드해서 진행.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:?}"
LOG="$TOPANEU_ROOT/experiments/d730_chain.log"
log() { echo "[d730-screen $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

STEP8_PID="${STEP8_WAIT_PID:?}"
log "STEP8(fold0 250ep, PID $STEP8_PID) 종료 대기 — 500ep/5fold는 스킵, 250ep 스크리닝만"
while kill -0 "$STEP8_PID" 2>/dev/null; do sleep 30; done

D730_F0_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D730_binary_vesselcrop_417/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP8 완료. D730_fold0_250ep_Dice=${D730_F0_DICE:-N/A}  vs  D720_fold0_250ep_기준=0.3305"
log "다음: D740(병변스케일crop) 빌드 + fold0 250ep 실행 대기 -> 세 후보 비교 후 5fold 결정"
