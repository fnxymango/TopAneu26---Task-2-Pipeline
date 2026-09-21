#!/usr/bin/env bash
# P3 선행 학습 — fold0 게이트 판정 전에 GPU1 에서 fold1, fold2 를 미리 돌린다 (2026-08-28).
#
# 투기적(speculative) 작업이다. fold0 이 게이트에서 떨어지면 이 체인은 죽여도 된다.
#   kill 방법: experiments/p3_gpu1.pid 의 PID 를 kill -TERM
# fold0 과 **같은 nnUNet_results 로** 떨어뜨려서 나중에 5폴드 확대 시 그대로 이어붙는다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
EXP=P3_pjh3cls_resencl_iso04_f0; PLANS=nnUNetResEncUNetLPlans722iso04; TR=nnUNetTrainer_250epochs
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
export nnUNet_results=$E/$EXP/results TOPANEU_ROOT=$R
cd $S; ST=$E/p3_status.log
log(){ echo "[P3-gpu1 $(TZ=Asia/Seoul date +'%m-%d %H:%M')] $*" | tee -a "$ST"; }
echo $$ > $E/p3_gpu1.pid

D="$nnUNet_results/Dataset722_TopAneuPjh3cls417/${TR}__${PLANS}__3d_fullres"
for f in 1 2; do
  if [ -f "$D/fold_${f}/checkpoint_final.pth" ]; then log "fold${f} 이미 완료 — 건너뜀"; continue; fi
  log "fold${f} 학습 시작 (GPU1, 선행)"
  CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_train" 722 3d_fullres $f -p "$PLANS" -tr "$TR" \
    > "$E/p3_train_f${f}.log" 2>&1
  if [ -f "$D/fold_${f}/checkpoint_best.pth" ]; then
    log "fold${f} 학습 완료 · $(du -h "$D/fold_${f}/checkpoint_best.pth"|cut -f1)"
  else
    log "★fold${f} 학습 실패 — 체크포인트 없음"; exit 1
  fi
done
log "GPU1 선행 학습 종료 (fold1, fold2)"
