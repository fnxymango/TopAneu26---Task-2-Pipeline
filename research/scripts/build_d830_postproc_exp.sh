#!/usr/bin/env bash
# D830 = D800 class-weighted(500ep, fold0) 예측에 후처리(close+prune+adj+endpoint)를 적용한 결과를
# 별도 실험으로 분리 저장한다. 학습은 없다 — 순수 후처리 파생 실험.
#
# 후처리 예측은 D800 실험 폴더에서 D830으로 **이동**(사본 중복 방지)하고,
# 리더보드/summary가 다른 실험과 같은 잣대로 비교되도록 nnU-Net 자체 평가기
# (nnUNetv2_evaluate_folder)로 summary.json을 생성한다.
set -euo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d830_build.log"

SRC_EXP="$TOPANEU_ROOT/experiments/D800_vessel_skelrec_resencm_417_500ep_classweighted"
DST_EXP="$TOPANEU_ROOT/experiments/D830_vessel_classweighted_postproc_417"
DS="Dataset800_TopAneuVessel417"
TRAINER_DIR="nnUNetTrainerSkeletonRecallNoMirroringClassWeighted_500ep_postproc__nnUNetResEncUNetMPlans__3d_fullres"
VALDST="$DST_EXP/results/$DS/$TRAINER_DIR/fold_0/validation"

export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"

log() { echo "[d830-build $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

PP_SRC="$SRC_EXP/results/validation_postprocessed"
[ -d "$PP_SRC" ] || { log "ABORT: 후처리 결과 폴더 없음 — $PP_SRC"; exit 1; }
NSRC=$(ls "$PP_SRC"/*.nii.gz 2>/dev/null | wc -l)
log "소스 후처리 예측 $NSRC 케이스 — $PP_SRC"

# ---------------------------------------------------------------- 1) 이동
mkdir -p "$VALDST"
mv "$PP_SRC"/*.nii.gz "$VALDST"/
rmdir "$PP_SRC" 2>/dev/null || true
log "예측 이동 완료 -> $VALDST ($(ls "$VALDST"/*.nii.gz | wc -l) 케이스)"

# ---------------------------------------------------------------- 2) val42 GT 심볼릭 폴더
GT_ALL="$NNUNET_BASE/nnUNet_preprocessed/$DS/gt_segmentations"
GT_VAL="$DST_EXP/gt_val42"
mkdir -p "$GT_VAL"
for f in "$VALDST"/*.nii.gz; do
  b=$(basename "$f")
  [ -e "$GT_VAL/$b" ] || ln -s "$GT_ALL/$b" "$GT_VAL/$b"
done
log "GT 심볼릭 $(ls "$GT_VAL" | wc -l)개 준비 (val42만)"

# ---------------------------------------------------------------- 3) nnU-Net 자체 평가기로 summary.json
DJ="$nnUNet_raw/$DS/dataset.json"
PF="$NNUNET_BASE/nnUNet_preprocessed/$DS/nnUNetResEncUNetMPlans.json"
log "nnUNetv2_evaluate_folder 실행 (내장 validation과 동일 지표)"
"$ENVBIN/nnUNetv2_evaluate_folder" "$GT_VAL" "$VALDST" -djfile "$DJ" -pfile "$PF" -np 4 >> "$LOG" 2>&1
log "summary.json 생성: $VALDST/summary.json"

# ---------------------------------------------------------------- 4) 부수 산출물
cp "$SCRIPTS/postproc_params.json" "$DST_EXP/postproc_params.json"
cp "$TOPANEU_ROOT/experiments/d800_classweighted_postproc.log" "$DST_EXP/postproc_apply.log"
log "postproc_params.json / postproc_apply.log 사본 저장"

log "완료"
