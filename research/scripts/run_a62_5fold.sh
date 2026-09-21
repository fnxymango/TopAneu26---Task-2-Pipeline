#!/usr/bin/env bash
# A6-2(adaptive norm + TverskyTopkCE) fold 1~4 학습 — 3/5 다수결 앙상블용.
#
# 왜 A6-2인가 (2026-08-14 c7 스윕 근거):
#   거리 게이팅+크기필터로 A6-2의 FP 152 -> 53까지 무료로 떨어뜨렸고 민감도는 0.814 그대로다.
#   같은 처리 후 A5-2는 0.721 @ 12FP. 즉 A6-2가 민감도 천장이 높고, 남은 53 FP는 전부
#   예측 혈관 3mm 이내(=해부학적으로 그럴듯한 위치)라 후처리로는 더 못 줄인다.
#   fold 간 재현되지 않는 이 잔여 FP를 걷어내는 게 다수결의 역할.
#
# split 안전성 (사용자 지시 2026-08-14 "val도 누출 0으로, train에서만 폴드 나눠"):
#   make_splits_417.py --train-only 로 재생성 — fold1~4를 공식 train 292 안에서만 나눈다.
#   확인 결과 5개 fold 전부 **val 42 누출 0 / test 83 누출 0**, fold0은 공식 split 그대로 불변.
#   덕분에 앙상블 멤버 5개 전원이 val 42를 학습에 쓰지 않아, val이 깨끗한 선택셋으로 남는다
#   -> 투표수(3/5)·후처리 임계를 val에서 고르고 test 83은 마지막에 한 번만 만진다.
#   대가는 fold1~4의 학습 케이스가 261 -> 219~220으로 줄어드는 것.
#
# GPU 2장이라 2개씩 두 라운드. 라운드당 ~15시간, 총 ~30시간.
# 중단: 이 스크립트 PID kill -> 각 fold는 checkpoint_latest.pth 에서 --c 로 재개 가능.
set -uo pipefail
R="${TOPANEU_ROOT:?}"
S="$R/code/sblee/nnunet/scripts"
LOG="$R/experiments/a62_5fold.log"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
PLANS="-p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE"

log() { echo "[a62-5fold $(date -u +'%m-%d %H:%M:%S')] $*" >> "$LOG"; }

run_fold() {  # $1=fold $2=gpu
  local f=$1 g=$2
  log "fold$f 시작 (GPU$g)"
  GPU="$g" NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$S/run_experiment.sh" 720 3d_fullres "$f" \
    "A6-2_resencl_adaptivenorm_topk_417_f$f" $PLANS >> "$LOG" 2>&1
  log "fold$f 종료 status=$?"
}

log "=========================================================="
log "A6-2 fold 1~4 시작 — 라운드1: fold1(GPU0) + fold2(GPU1)"
run_fold 1 0 & P1=$!
run_fold 2 1 & P2=$!
wait $P1; wait $P2
log "라운드1 완료 — 라운드2: fold3(GPU0) + fold4(GPU1)"
run_fold 3 0 & P3=$!
run_fold 4 1 & P4=$!
wait $P3; wait $P4
log "A6-2 5-fold 학습 전부 완료"
