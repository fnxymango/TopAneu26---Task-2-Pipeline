#!/usr/bin/env bash
# GPU0 체인 — 멱등(idempotent) 재개 버전. supervisor가 죽은 체인을 되살릴 때 쓴다.
#   이미 끝난 단계는 건너뛰고, 학습이 중단됐으면 checkpoint_latest에서 --c로 이어받는다.
# 단계: 1) A6-2 (ResEncL + adaptive norm + topk, 공식 split fold0)
#       2) A4(lesionscale) 전체볼륨 평가
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/gpu0_chain_a6.log"
E="$TOPANEU_ROOT/experiments"
EXP="A6-2_resencl_adaptivenorm_topk_417_f0"
export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
log() { echo "[gpu0-chain $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

done_train() { grep -q "finished with status 0" "$E/$EXP/train.log" 2>/dev/null; }
has_ckpt()   { ls "$E/$EXP"/results/*/*/fold_0/checkpoint_latest.pth >/dev/null 2>&1; }

if done_train; then
  log "STEP1 이미 완료 — 건너뜀"
else
  CONT=""
  if has_ckpt; then CONT="--c"; log "STEP1 재개: checkpoint_latest에서 --c"; else log "STEP1 시작: $EXP (공식 split fold0)"; fi
  GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 "$EXP" \
    -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE $CONT >> "$LOG" 2>&1
  D=$(grep -h "Mean Validation Dice" "$E/$EXP"/results/*/*/fold_0/training_log_*.txt 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  log "STEP1 종료 adaptive+topk Dice=${D:-N/A}"
  "$ENVBIN/python" "$SCRIPTS/make_summary.py" "$E/$EXP" >> "$LOG" 2>&1
  "$ENVBIN/python" "$SCRIPTS/make_leaderboard.py" >> "$LOG" 2>&1
fi

if [ -f "$TOPANEU_ROOT/code/sblee/nnunet/analysis/fullvolume_eval_A4_lesionscale_fullvol.json" ]; then
  log "STEP2 이미 완료 — 건너뜀"
else
  log "STEP2: A4(lesionscale) 전체볼륨 평가"
  bash "$SCRIPTS/run_a4_lesionscale_fullvolume.sh" >> "$LOG" 2>&1
fi
log "GPU0 체인 완료"
