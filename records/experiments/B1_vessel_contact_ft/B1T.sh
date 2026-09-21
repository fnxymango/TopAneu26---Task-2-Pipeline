#!/usr/bin/env bash
# B1T — B1 fine-tune 두 팔(GPU0 ctrl λ0 · GPU1 contact λ1) → val 41 추론·후처리·그래프 → 관문 V 보고. 규칙: PLAN.md
B=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/B1_vessel_contact_ft
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
export nnUNet_raw=$TOPANEU_ROOT/nnunet/nnUNet_raw nnUNet_preprocessed=/mnt/hdd/sblee/topaneu_archive/nnUNet_preprocessed nnUNet_results=$B/results
export nnUNet_compile=f nnUNet_n_proc_DA=8
exec 9>"$B/b1t.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b1t] $*" | tee -a "$D/STATUS.log"; }
log "대기 · .done_b1p"
while [ ! -f "$B/.done_b1p" ]; do sleep 60; done
log "압축 해제(두 학습이 같은 폴더를 동시에 풀지 않도록 먼저)"
"$PY" -c "from nnunetv2.training.dataloading.utils import unpack_dataset; unpack_dataset('$nnUNet_preprocessed/Dataset801_TopAneuVesselFT/nnUNetPlans_3d_fullres', True, False, 8)" || { log "압축 해제 실패"; exit 1; }
log "학습 시작 · ctrl GPU0 · contact GPU1 · 100 epoch"
CUDA_VISIBLE_DEVICES=0 $HOME/miniconda3/envs/sbaneu2/bin/nnUNetv2_train 801 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerVesselFT_ctrl > "$B/train_ctrl.log" 2>&1 &
sleep 120
CUDA_VISIBLE_DEVICES=1 $HOME/miniconda3/envs/sbaneu2/bin/nnUNetv2_train 801 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerVesselFT_contact > "$B/train_contact.log" 2>&1 &
wait
for a in ctrl contact; do
  f=$B/results/Dataset801_TopAneuVesselFT/nnUNetTrainerVesselFT_${a}__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_final.pth
  [ -f "$f" ] || { log "학습 실패 · $a checkpoint_final 없음"; exit 1; }
done
log "학습 끝 · val 추론"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$B/b1_val.py" predict ctrl > "$B/val_ctrl.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$B/b1_val.py" predict contact > "$B/val_contact.log" 2>&1 &
wait
"$PY" -u "$B/b1_val.py" post ctrl >> "$B/val_ctrl.log" 2>&1 && "$PY" -u "$B/b1_val.py" post contact >> "$B/val_contact.log" 2>&1 || { log "후처리 실패"; exit 1; }
while [ ! -f "$B/.done_b1base" ]; do sleep 60; done
"$PY" "$B/b1_val.py" report > "$B/RESULTS_B1_V.md" 2>&1 && touch "$B/.done_b1v"
log "관문 V 끝 → $(grep -o '관문 V[^*]*' $B/RESULTS_B1_V.md)"
