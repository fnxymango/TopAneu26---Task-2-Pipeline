#!/usr/bin/env bash
# D800 STEPA(fold0 500ep) 완료 감지 -> STEPB(5-fold, fold1~4) 시작되기 전에 최상위
# 체인 wrapper를 죽여서 5-fold를 취소하고 -> fold0 val 예측에 후처리(close+prune+adj+endpoint)
# 적용 및 전/후 비교 (사용자 지시 2026-08-10: "D800은 일단 5폴드 제외해").
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d800_stepa_postproc.log"
EXP_DIR="$TOPANEU_ROOT/experiments/V2-2_vessel_skelrec_resencm_417_500ep"
TRAIN_LOG="$EXP_DIR/train.log"
CHAIN_WRAPPER_PID="${CHAIN_WRAPPER_PID:-2875513}"
mkdir -p "$TOPANEU_ROOT/experiments"

export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"

log() { echo "[d800-stepa-postproc $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "STEPA(fold0) 완료 대기 -> $TRAIN_LOG 에서 'finished with status' 모니터링"
while true; do
  if [ -f "$TRAIN_LOG" ] && grep -q "\[run_experiment\] finished with status" "$TRAIN_LOG"; then
    break
  fi
  sleep 10
done
STATUS_LINE=$(grep "\[run_experiment\] finished with status" "$TRAIN_LOG" | tail -1)
log "STEPA 완료 감지: $STATUS_LINE"

# ---------------------------------------------------------------- 5-fold(STEPB) 취소
log "STEPB(5-fold) 취소 시도: wrapper PID $CHAIN_WRAPPER_PID kill"
kill "$CHAIN_WRAPPER_PID" 2>/dev/null && log "  wrapper kill 신호 전송" || log "  wrapper 이미 종료됨(또는 못찾음)"
sleep 5
# 혹시 그 사이 fold1이 이미 떠버렸으면 같이 정리
FOLD1_PIDS=$(pgrep -f "nnUNetv2_train 800 3d_fullres 1" || true)
if [ -n "$FOLD1_PIDS" ]; then
  log "  fold1이 이미 시작됨(PID $FOLD1_PIDS) -> 같이 kill"
  kill $FOLD1_PIDS 2>/dev/null || true
fi
log "STEPB 취소 처리 완료"

# ---------------------------------------------------------------- 후처리 fit + apply + eval
log "postprocess fit (Dataset800/417 GT)"
"$ENVBIN/python" "$SCRIPTS/postprocess_vessel.py" fit >> "$LOG" 2>&1

VAL_DIR=$(find "$EXP_DIR/results" -type d -path "*fold_0/validation" | head -1)
if [ -z "$VAL_DIR" ]; then
  log "ABORT: fold0 validation 예측 폴더를 못 찾음"
  exit 1
fi
log "fold0 validation 예측 폴더: $VAL_DIR"

PP_OUT="$EXP_DIR/results/validation_postprocessed"
log "postprocess apply (close+prune+adj+endpoint) -> $PP_OUT"
"$ENVBIN/python" "$SCRIPTS/postprocess_vessel.py" apply "$VAL_DIR" "$PP_OUT" >> "$LOG" 2>&1

log "postprocess eval (전/후 비교)"
"$ENVBIN/python" "$SCRIPTS/postprocess_vessel.py" eval "$VAL_DIR" "$PP_OUT" >> "$LOG" 2>&1

log "D800 fold0 후처리 파이프라인 전체 완료"
