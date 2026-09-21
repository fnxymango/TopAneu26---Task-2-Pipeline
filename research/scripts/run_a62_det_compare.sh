#!/usr/bin/env bash
# GPU0 — 검출기 교체 대조군 (사용자 지시 2026-08-14).
#
# 왜: GPU1 체인(run_gpu1_chain_c1.sh)은 08-13 16:30에 짜여서 A5-2(plain z)만 쓸 수 있었다.
#     A6-2(adaptive norm)는 08-14 07:37에 끝나 16분 차이로 체인에 못 들어갔다.
#     두 모델은 GT 음성 케이스에서 오경보가 크게 다르므로(A5-2 5/11 · 7106복셀,
#     A6-2 2/11 · 764복셀), C단계 위치분류 점수가 검출기 위양성에 얼마나 끌리는지
#     답하려면 같은 C 방법에 검출기만 바꾼 대조군이 필요하다.
#
# 레버는 하나만: 동맥류 검출기 A5-2 -> A6-2. 혈관(V4-2)·후처리(V5)·C방법은 GPU1 체인과 동일.
# 출력은 *_a62 접미사로 분리해 돌고 있는 GPU1 체인의 산출물을 건드리지 않는다.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
E="$TOPANEU_ROOT/experiments"
WORK="$E/_c1_realpred"
LOG="$E/a62_det_compare.log"
EXP_A62="A6-2_resencl_adaptivenorm_topk_417_f0"

export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
log() { echo "[a62-cmp $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

predict() {  # $1=split
  # 주의: `local A=.. B="..$A.."` 한 줄로 쓰면 bash가 local 실행 전에 전부 전개해
  #       set -u 에서 unbound variable 로 죽는다. 반드시 분리할 것.
  local SP="$1"
  local OUT="$WORK/aneu_${SP}_a62"
  if [ "$(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)" -gt 0 ] && [ -f "$OUT/.done" ]; then
    log "STEP1($SP) 이미 완료 — 건너뜀"; return 0
  fi
  log "STEP1($SP): A6-2 동맥류 추론 -> $(basename "$OUT")"
  nnUNet_results="$E/$EXP_A62/results" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
    -i "$WORK/in_$SP" -o "$OUT" -d 720 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTverskyTopkCE -p nnUNetResEncUNetLPlansAdaptive -chk checkpoint_best.pth \
    --disable_tta -npp 2 -nps 2 >> "$LOG" 2>&1
  local S=$?
  log "STEP1($SP) 종료 status=$S  파일 $(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)개"
  [ $S -eq 0 ] && touch "$OUT/.done"
  return $S
}

# ---- val: 추론 + 병변단위 평가 (A5-2와 같은 잣대) ----
predict val || { log "ABORT: val 추론 실패"; exit 1; }
log "STEP2(val): 병변단위 평가 (민감도 / 케이스당 위양성)"
"$ENVBIN/python" "$SCRIPTS/eval_d740_fullvolume.py" \
  "$WORK/aneu_val_a62" A6-2_adaptive_aneu_val >> "$LOG" 2>&1
log "STEP2(val) 종료 status=$?"

# ---- test: in_test는 GPU1 체인이 이미 만들어 둠. 없으면 대기 ----
for i in $(seq 1 60); do
  [ "$(ls "$WORK/in_test"/*.nii.gz 2>/dev/null | wc -l)" -eq 83 ] && break
  log "in_test 대기중 ($i/60)"; sleep 60
done
predict test || { log "ABORT: test 추론 실패"; exit 1; }

log "완료 — 산출물: aneu_val_a62 / aneu_test_a62,"
log "        analysis/fullvolume_eval_A6-2_adaptive_aneu_val.json"
log "        다음: GPU1 체인 종료 후 d900/d910을 --aneurysm-pred-dir aneu_*_a62 로 재실행"
