#!/usr/bin/env bash
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee; D=$R/experiments/D1_newdata; ST=$D/STATUS.log
export TOPANEU_ROOT=$R nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
export PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet:${PYTHONPATH:-}
EXP=P5_newdata_detector
CK=$R/experiments/$EXP/results/Dataset722_TopAneuPjh3cls417/nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans722iso04__3d_fullres
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][det-gpu0] $*" | tee -a "$ST"; }
echo $$ > "$D/det_gpu0.pid"
for f in 0 1 2; do
  if [ -f "$CK/fold_$f/checkpoint_final.pth" ]; then log "fold$f 이미 완료"; continue; fi
  CONT=""
  if [ -f "$CK/fold_$f/checkpoint_latest.pth" ]; then CONT="--c"; log "fold$f 중단지점부터 이어받음(--c)"; else log "fold$f 학습 시작"; fi
  GPU=0 NPROC=1 ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin TOPANEU_ROOT=$R \
    bash "$R/code/sblee/nnunet/scripts/run_experiment.sh" 722 3d_fullres $f "$EXP" \
    -p nnUNetResEncUNetLPlans722iso04 -tr nnUNetTrainer_250epochs $CONT >> "$D/det_f$f.log" 2>&1
  if [ -f "$CK/fold_$f/checkpoint_final.pth" ]; then
    log "fold$f 완료 · $(grep -oE 'Mean Validation Dice: [0-9.]+' "$D/det_f$f.log" | tail -1)"
  else
    log "★fold$f 실패 — $D/det_f$f.log"; exit 1
  fi
done
log "GPU0 담당 폴드(0 1 2) 전부 완료"
touch "$D/.done_det_gpu0"
