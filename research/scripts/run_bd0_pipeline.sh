#!/usr/bin/env bash
# D600 batch_dice=False 재학습 이후 전 과정 자동 실행 (세션 독립).
#   1) 후처리 파라미터(fit) 대기 → 기존 예측에 후처리 적용 + 전/후 채점 (후처리 자체 효과 측정)
#   2) 재학습 완료 대기 → val 예측(메모리안전 -nps1) → 후처리 → 전/후 채점
#   3) 후처리된 새 예측으로 skeleton 이미지 재생성
set -uo pipefail
BASE=/home/user/TopAneu/seg/sblee/nnunet
PY=/home/user/anaconda3/envs/sbaneu2/bin
OLD=$BASE/experiments/D600_vessel_skelrec_resencm_250ep
NEW=$BASE/experiments/D600_vessel_skelrec_resencm_250ep_bd0
REPORT=$NEW/pipeline.log

export nnUNet_raw=$BASE/nnUNet_raw
export nnUNet_preprocessed=$BASE/nnUNet_preprocessed
export CUDA_VISIBLE_DEVICES=1

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$REPORT"; }
mkdir -p "$NEW"

# ---- 1) 후처리 파라미터 대기 → 기존(batch_dice=True) 예측에 적용해 효과 측정 ----
say "후처리 파라미터(fit) 대기..."
until [ -f "$BASE/scripts/postproc_params.json" ]; do sleep 30; done
say "fit 완료. 기존 예측에 후처리 적용 (후처리 단독 효과 측정)"
"$PY/python" -u "$BASE/scripts/postprocess_vessel.py" apply "$OLD/val_predict_out" "$OLD/val_predict_out_pp" 2>&1 | tee -a "$REPORT"
"$PY/python" -u "$BASE/scripts/postprocess_vessel.py" eval "$OLD/val_predict_out" "$OLD/val_predict_out_pp" 2>&1 | tee -a "$REPORT"

# ---- 2) 재학습 완료 대기 ----
say "재학습 완료 대기 (checkpoint_final.pth)"
CKDIR=$NEW/results/Dataset600_TopAneuVessel/nnUNetTrainerSkeletonRecallNoMirroring_250ep__nnUNetResEncUNetMPlansBD0__3d_fullres/fold_0
until [ -f "$CKDIR/checkpoint_final.pth" ]; do
  pgrep -u "$USER" -f "nnUNetv2_train 600 3d_fullres 0 -p nnUNetResEncUNetMPlansBD0" >/dev/null || {
    sleep 60
    [ -f "$CKDIR/checkpoint_final.pth" ] || { say "학습 프로세스가 사라졌는데 최종 체크포인트 없음 — 중단"; exit 1; }
  }
  sleep 120
done
say "학습 완료. val 15 예측 시작 (-nps 1, 메모리안전)"

export nnUNet_results=$NEW/results
mkdir -p "$NEW/val_predict_in"
cp -a "$OLD/val_predict_in/." "$NEW/val_predict_in/" 2>/dev/null
"$PY/nnUNetv2_predict" -i "$NEW/val_predict_in" -o "$NEW/val_predict_out" \
  -d 600 -c 3d_fullres -f 0 -p nnUNetResEncUNetMPlansBD0 \
  -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep -npp 1 -nps 1 2>&1 | tail -20 | tee -a "$REPORT"

say "새 예측에 후처리 적용"
"$PY/python" -u "$BASE/scripts/postprocess_vessel.py" apply "$NEW/val_predict_out" "$NEW/val_predict_out_pp" 2>&1 | tee -a "$REPORT"
say "채점: 신규 raw vs 신규 후처리"
"$PY/python" -u "$BASE/scripts/postprocess_vessel.py" eval "$NEW/val_predict_out" "$NEW/val_predict_out_pp" 2>&1 | tee -a "$REPORT"
say "채점: 기존 모델(batch_dice=True) 대비"
"$PY/python" -u "$BASE/scripts/postprocess_vessel.py" eval "$OLD/val_predict_out" "$NEW/val_predict_out_pp" 2>&1 | tee -a "$REPORT"

# ---- 3) skeleton 이미지 재생성 ----
say "skeleton 이미지 생성 (후처리된 새 예측)"
"$PY/python" -u "$BASE/scripts/skeleton_from_predictions.py" \
  "$NEW/val_predict_out_pp" "$NEW/val_skeletons" 2>&1 | tail -20 | tee -a "$REPORT"
say "파이프라인 완료"
