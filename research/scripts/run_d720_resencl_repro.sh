#!/usr/bin/env bash
# 외부에서 들은 "ResEncL + binary aneurysm + 5fold majority vote(3-of-5), plain vs topk -> Dice 0.82" 재현.
# 1단계: fold0만 plain(stock nnUNetTrainer_250epochs) vs topk(project nnUNetTrainerTverskyTopkCE, 자체 250ep default)
# 비교 스크리닝. 사용자 지시(2026-08-10): topk=프로젝트 커스텀 TverskyTopkCE, 250ep, fold0 먼저 보고 5fold 여부 결정.
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d720_resencl_repro_chain.log"
mkdir -p "$TOPANEU_ROOT/experiments"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export TOPANEU_DATA="${TOPANEU_DATA:-$TOPANEU_ROOT/dataset/TopAneu}"

log() { echo "[d720-resencl-repro $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

WAIT_PID="${WAIT_PID:-2898972}"
if [ "${SKIP_WAIT:-0}" != "1" ]; then
  log "GPU0 선점 프로세스(PID $WAIT_PID, D740) 종료 대기"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  log "GPU0 확보"
fi

log "STEP1 시작: Dataset720에 ResEncL plan+preprocess (nnUNetPlannerResEncL, gpu_memory_target=20GB)"
# 기본 24GB 타겟은 이 카드(RTX A5000 24564MiB)에서 여유가 거의 없음 -> D800(36class) resencl이
# epoch0에 바로 OOM난 전례(2026-08-08 16:27) 있어 20GB로 낮춰 안전마진 확보 (사용자 지시 2026-08-10).
"$ENVBIN/nnUNetv2_plan_and_preprocess" -d 720 -pl nnUNetPlannerResEncL -gpu_memory_target 20 \
  -overwrite_plans_name nnUNetResEncUNetLPlans -c 3d_fullres -np 4 >> "$LOG" 2>&1
STEP1_STATUS=$?
[ $STEP1_STATUS -eq 0 ] || die "STEP1 실패 status=$STEP1_STATUS"
log "STEP1 완료"

log "STEP2 시작: D720 ResEncL fold0 plain (nnUNetTrainer_250epochs)"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_plain_417_f0 \
  -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_250epochs >> "$LOG" 2>&1
STEP2_STATUS=$?
PLAIN_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_plain_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP2 종료 status=$STEP2_STATUS plain_fold0_Dice=${PLAIN_DICE:-N/A}"

log "STEP3 시작: D720 ResEncL fold0 topk (nnUNetTrainerTverskyTopkCE, 자체 250ep default)"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_topk_417_f0 \
  -p nnUNetResEncUNetLPlans -tr nnUNetTrainerTverskyTopkCE >> "$LOG" 2>&1
STEP3_STATUS=$?
TOPK_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_topk_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP3 종료 status=$STEP3_STATUS topk_fold0_Dice=${TOPK_DICE:-N/A}"

log "plain-z 스크리닝 완료. plain=${PLAIN_DICE:-N/A}  topk=${TOPK_DICE:-N/A}  (외부소스 claim=0.82, D720 기존 최고=0.4157)"

# ---------------------------------------------------------------- STEP4: adaptive-norm plans 생성
# 사용자 지시(2026-08-10): adaptive norm 재검토. 기존 D521(구98케이스+구auto-split) 결과(0.4244)로
# "adaptive 열세" 판정났던 게 실은 canonical split으로 재검증 안 된 상태였음(docs/results_2026-07-21
# _canon_loss_sweep.md: "adaptive/521은 정식 split 재실행 안 함") -> 417케이스+ResEncL로 공정 재검증.
log "STEP4 시작: adaptive-norm plans 생성 (TopAneuAdaptiveNorm, plain-z ResEncL plan 복제+정규화만 교체)"
"$ENVBIN/python" - <<'PYEOF' >> "$LOG" 2>&1
import json, os
root = os.environ["TOPANEU_ROOT"]
pp = f"{root}/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417"
d = json.load(open(f"{pp}/nnUNetResEncUNetLPlans.json"))
d["plans_name"] = "nnUNetResEncUNetLPlansAdaptive"
c = d["configurations"]["3d_fullres"]
c["data_identifier"] = "nnUNetResEncUNetLPlansAdaptive_3d_fullres"
c["normalization_schemes"] = ["TopAneuAdaptiveNorm"]
json.dump(d, open(f"{pp}/nnUNetResEncUNetLPlansAdaptive.json", "w"), indent=4)
print(f"[adaptive-plan] 저장: {pp}/nnUNetResEncUNetLPlansAdaptive.json")
PYEOF
STEP4A_STATUS=$?
[ $STEP4A_STATUS -eq 0 ] || die "STEP4 adaptive plan 생성 실패 status=$STEP4A_STATUS"

log "STEP4 계속: adaptive-norm으로 실제 전처리(preprocess)"
"$ENVBIN/nnUNetv2_preprocess" -d 720 -plans_name nnUNetResEncUNetLPlansAdaptive -c 3d_fullres -np 4 >> "$LOG" 2>&1
STEP4B_STATUS=$?
[ $STEP4B_STATUS -eq 0 ] || die "STEP4 preprocess 실패 status=$STEP4B_STATUS"
log "STEP4 완료: adaptive-norm 전처리 완료"

log "STEP5 시작: D720 ResEncL(adaptive-norm) fold0 plain"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_adaptivenorm_plain_417_f0 \
  -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainer_250epochs >> "$LOG" 2>&1
STEP5_STATUS=$?
ADAPT_PLAIN_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_adaptivenorm_plain_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP5 종료 status=$STEP5_STATUS adaptive_plain_fold0_Dice=${ADAPT_PLAIN_DICE:-N/A} (plain-z 기준=${PLAIN_DICE:-N/A})"

log "STEP6 시작: D720 ResEncL(adaptive-norm) fold0 topk"
GPU=0 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 D720_resencl_adaptivenorm_topk_417_f0 \
  -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE >> "$LOG" 2>&1
STEP6_STATUS=$?
ADAPT_TOPK_DICE=$(grep "Mean Validation Dice" "$TOPANEU_ROOT/experiments/D720_resencl_adaptivenorm_topk_417_f0/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
log "STEP6 종료 status=$STEP6_STATUS adaptive_topk_fold0_Dice=${ADAPT_TOPK_DICE:-N/A} (plain-z 기준=${TOPK_DICE:-N/A})"

log "전체 스크리닝 완료 4종 비교: plainZ+plain=${PLAIN_DICE:-N/A}  plainZ+topk=${TOPK_DICE:-N/A}  adaptive+plain=${ADAPT_PLAIN_DICE:-N/A}  adaptive+topk=${ADAPT_TOPK_DICE:-N/A}  (외부소스 claim=0.82, D720 기존 최고=0.4157) -> 5-fold 진행 여부는 사용자 결정 대기"
