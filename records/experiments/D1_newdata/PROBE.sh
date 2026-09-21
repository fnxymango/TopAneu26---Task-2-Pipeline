#!/usr/bin/env bash
# 미검출 병변 자리에 검출기 확률이 조금이라도 있었나 — argmax 아래 신호 유무 확인
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee; E=$R/experiments; P=$E/_c1_realpred; D=$E/D1_newdata
OUT=$D/probe; mkdir -p $OUT/in $OUT/pred
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
for c in topaneu_center1_mr_024 topaneu_center1_mr_917 topaneu_center2_mr_086 topaneu_center5_mr_061 topaneu_center1_mr_092; do
  ln -sf "$P/in_test_722/${c}_0000.nii.gz" "$OUT/in/${c}_0000.nii.gz"
done
nnUNet_results="$E/E9_ensemble10/results" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
  -i "$OUT/in" -o "$OUT/pred" -d 722 -c 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 \
  -p nnUNetResEncUNetLPlans722iso04 -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
  --disable_tta --save_probabilities -npp 2 -nps 2 > "$OUT/predict.log" 2>&1
echo "rc=$?"; touch "$D/.done_probe"
