#!/usr/bin/env bash
# run_d720_resencl_repro.sh의 STEP4~6 재개 스크립트.
# 원 체인은 2026-08-11 22:48 STEP4(adaptive-norm preprocess) 도중 디스크 부족(264/417, status 137,
# BrokenPipe)으로 ABORT. Dataset800 전처리(126G)를 /mnt/hdd로 이동+심링크하여 / 여유 28G->154G 확보 후 재개.
# STEP1~3(plan/preprocess, plain fold0=0.3638, topk fold0=0.5273)는 완료됨. 여기선 STEP4~6만.
# 변경점: preprocess -np 4 -> 2 (워커 RAM 부담 완화). TOPANEU_ROOT를 실제 트리(B)로 명시.
set -uo pipefail

# 주의: $HOME/topaneu_sblee 는 빈 껍데기 트리(A). 실제 데이터/코드는 아래 경로(B).
TOPANEU_ROOT="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d720_resencl_repro_resume_step4.log"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export TOPANEU_DATA="${TOPANEU_DATA:-$TOPANEU_ROOT/dataset/TopAneu}"
PP="$nnUNet_preprocessed/Dataset720_TopAneuBinary417"

log() { echo "[d720-resume $(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

log "=== STEP4~6 재개 시작 (Dataset800 HDD 이동으로 / 확보 후, -np 2) ==="
log "디스크 현황: $(df -h / | awk 'NR==2{print $4" free ("$5")"}')"

# ---- STEP4A: adaptive plan JSON (이미 생성돼 있음 -> 재사용) --------------------
[ -f "$PP/nnUNetResEncUNetLPlansAdaptive.json" ] || die "adaptive plan JSON 없음: $PP/nnUNetResEncUNetLPlansAdaptive.json"
log "STEP4A: adaptive plan JSON 확인됨(재사용)"

# ---- STEP4B: adaptive-norm 전처리 (clean 후 -np 2) -----------------------------
if [ -d "$PP/nnUNetResEncUNetLPlansAdaptive_3d_fullres" ]; then
  log "기존(부분) adaptive 데이터 폴더 제거하여 clean preprocess 보장"
  rm -rf "$PP/nnUNetResEncUNetLPlansAdaptive_3d_fullres"
fi
log "STEP4B 시작: nnUNetv2_preprocess -d 720 -plans_name nnUNetResEncUNetLPlansAdaptive -c 3d_fullres -np 2"
"$ENVBIN/nnUNetv2_preprocess" -d 720 -plans_name nnUNetResEncUNetLPlansAdaptive -c 3d_fullres -np 2 >> "$LOG" 2>&1
STEP4B_STATUS=$?
[ $STEP4B_STATUS -eq 0 ] || die "STEP4B preprocess 실패 status=$STEP4B_STATUS"
log "STEP4B 완료. / 여유: $(df -h / | awk 'NR==2{print $4}')  adaptive데이터=$(du -sh "$PP/nnUNetResEncUNetLPlansAdaptive_3d_fullres" 2>/dev/null | cut -f1)"

# ---- STEP5: adaptive-norm plain fold0 -----------------------------------------
log "STEP5 시작: D720 ResEncL(adaptive-norm) fold0 plain (nnUNetTrainer_250epochs)"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_adaptivenorm_plain_417_f0 \
  -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainer_250epochs >> "$LOG" 2>&1
STEP5_STATUS=$?
ADAPT_PLAIN_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_adaptivenorm_plain_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP5 종료 status=$STEP5_STATUS adaptive_plain_fold0_Dice=${ADAPT_PLAIN_DICE:-N/A} (plain-z 기준=0.3638)"

# ---- STEP6: adaptive-norm topk fold0 ------------------------------------------
log "STEP6 시작: D720 ResEncL(adaptive-norm) fold0 topk (nnUNetTrainerTverskyTopkCE)"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_adaptivenorm_topk_417_f0 \
  -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE >> "$LOG" 2>&1
STEP6_STATUS=$?
ADAPT_TOPK_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_adaptivenorm_topk_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP6 종료 status=$STEP6_STATUS adaptive_topk_fold0_Dice=${ADAPT_TOPK_DICE:-N/A} (plain-z topk 기준=0.5273)"

log "=== 재개 완료. 4종 fold0 비교: plainZ+plain=0.3638  plainZ+topk=0.5273  adaptive+plain=${ADAPT_PLAIN_DICE:-N/A}  adaptive+topk=${ADAPT_TOPK_DICE:-N/A}  (외부 claim=0.82, D720 기존최고=0.4157) -> 5-fold 진행 여부는 사용자 결정 대기 ==="
