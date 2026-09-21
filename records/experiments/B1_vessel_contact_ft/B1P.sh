#!/usr/bin/env bash
# B1P — Dataset801 전처리(재계획 금지 · Dataset800 plans 복사본) · 원본 train+val 332
B=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/B1_vessel_contact_ft
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
export nnUNet_raw=$TOPANEU_ROOT/nnunet/nnUNet_raw nnUNet_preprocessed=/mnt/hdd/sblee/topaneu_archive/nnUNet_preprocessed nnUNet_results=$B/results
exec 9>"$B/b1p.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b1p] $*" | tee -a "$D/STATUS.log"; }
log "B1 전처리 시작 · Dataset801 · 332케이스 · np 8"
$HOME/miniconda3/envs/sbaneu2/bin/nnUNetv2_preprocess -d 801 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres -np 8 && touch "$B/.done_b1p"
log "B1 전처리 끝 · $(ls $nnUNet_preprocessed/Dataset801_TopAneuVesselFT/nnUNetPlans_3d_fullres 2>/dev/null | wc -l) 파일 · done=$([ -f $B/.done_b1p ] && echo y || echo n)"
