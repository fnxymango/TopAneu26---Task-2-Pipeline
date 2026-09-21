#!/usr/bin/env bash
# D800 class-weighted 재학습(fold0, Dice 0.7793) 결과에 후처리 적용 + 전/후 비교.
# postproc_params.json은 GT에서 fit한 것이라 모델 무관 -> fit 재실행 없이 apply/eval만.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d800_classweighted_v2_postproc.log"
EXP="$TOPANEU_ROOT/experiments/V4-2_vessel_classweighted_417_500ep"
export TOPANEU_ROOT
export nnUNet_raw="$TOPANEU_ROOT/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$TOPANEU_ROOT/nnunet/nnUNet_preprocessed"
log() { echo "[d800-cw-postproc $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

VAL_DIR=$(find "$EXP/results" -type d -path "*fold_0/validation" | head -1)
[ -n "$VAL_DIR" ] || { log "ABORT: validation 폴더 못찾음"; exit 1; }
PP_OUT="$EXP/results/validation_postprocessed"

log "apply (close+prune+adj+endpoint) -> $PP_OUT"
"$ENVBIN/python" "$SCRIPTS/postprocess_vessel.py" apply "$VAL_DIR" "$PP_OUT" >> "$LOG" 2>&1

log "eval (전/후 비교)"
"$ENVBIN/python" "$SCRIPTS/postprocess_vessel.py" eval "$VAL_DIR" "$PP_OUT" >> "$LOG" 2>&1
log "완료"
