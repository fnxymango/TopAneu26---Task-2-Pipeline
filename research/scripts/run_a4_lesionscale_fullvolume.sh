#!/usr/bin/env bash
# D760(adaptive-norm) 학습 2종이 끝나면 -> D740(lesionscale) 모델로 정식 val 84케이스
# **전체볼륨** 추론 + 평가. (사용자 승인 2026-08-13)
#
# 배경: D740의 기존 Dice 0.5131은 crop window 106개에서 잰 값이라 전체볼륨에서 잰
# A5-2(0.5273)과 잣대가 다름. 또 crop 학습 -> 전체볼륨 추론이라 위양성 폭발 위험이
# 구조적으로 존재. D770(ResEncL+lesionscale 스택)에 투자하기 전에 이걸 먼저 확인한다.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/d740_fullvolume_eval.log"
E="$TOPANEU_ROOT/experiments"
D740="$E/A4_binary_lesionscale_417"
WORK="$TOPANEU_ROOT/experiments/_d740_fullvol_tmp"

export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$D740/results"
log() { echo "[d740-fullvol $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "D760(adaptive) 학습 2종 종료 대기"
# (체인에서 순서 보장하므로 대기 불필요)
log "GPU 확보 확인"
sleep 30

# ---- 입력 준비: 정식 val 84케이스 전체영상만 심볼릭으로 모음 ----
IN="$WORK/in"; OUT="$WORK/pred"
rm -rf "$WORK"; mkdir -p "$IN" "$OUT"
"$ENVBIN/python" - <<'PY' >> "$LOG" 2>&1
import json, os
from pathlib import Path
root = Path(os.environ["TOPANEU_ROOT"])
pre = root/"nnunet"/"nnUNet_preprocessed"/"Dataset720_TopAneuBinary417"
img = root/"nnunet"/"nnUNet_raw"/"Dataset720_TopAneuBinary417"/"imagesTr"
IN = root/"experiments"/"_d740_fullvol_tmp"/"in"
val = json.load(open(pre/"splits_final.json"))[0]["val"]
n=0
for cid in val:
    src = img/f"{cid}_0000.nii.gz"
    if src.exists():
        os.symlink(src, IN/f"{cid}_0000.nii.gz"); n+=1
print(f"[prep] val {len(val)}케이스 중 {n}개 링크")
PY

log "전체볼륨 sliding-window 추론 시작 (D740 checkpoint_best, TTA 끔)"
CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
  -i "$IN" -o "$OUT" -d 740 -c 3d_fullres -f 0 \
  -tr nnUNetTrainerTverskyTopkCE -p nnUNetPlans -chk checkpoint_best.pth \
  --disable_tta -npp 2 -nps 2 >> "$LOG" 2>&1
PRED_STATUS=$?
log "추론 종료 status=$PRED_STATUS"
[ $PRED_STATUS -eq 0 ] || { log "ABORT: 추론 실패"; exit 1; }

log "평가 (case Dice + 병변 검출 sensitivity + 케이스당 위양성)"
"$ENVBIN/python" "$SCRIPTS/eval_d740_fullvolume.py" "$OUT" D740_lesionscale_fullvol >> "$LOG" 2>&1
log "완료 — 비교대상: A5-2 plainz+topk=0.5273(전체볼륨), A4 crop내부평가=0.5131"
