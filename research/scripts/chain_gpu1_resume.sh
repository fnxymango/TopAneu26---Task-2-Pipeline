#!/usr/bin/env bash
# GPU1 체인 — 멱등(idempotent) 재개 버전. supervisor가 죽은 체인을 되살릴 때 쓴다.
#   이미 만들어진 산출물이 있으면 건너뛰고, 학습이 중단됐으면 --c로 이어받는다.
# 단계: 1) A5-2 (ResEncL + plain-z + topk, 공식 split fold0)
#       2) 동맥류 추론(val/test)  3) 혈관 추론(V4-2)  4) 혈관 후처리  5) C1/C2 평가
# 최종 성적은 test 83 기준(val 42는 fold0 검증셋이라 참고용).
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ANALYSIS="$TOPANEU_ROOT/code/sblee/nnunet/analysis"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
ENVBIN2="$HOME/miniconda3/envs/sbaneu2/bin"
LOG="$TOPANEU_ROOT/experiments/gpu1_chain_c1.log"
E="$TOPANEU_ROOT/experiments"
WORK="$E/_c1_realpred"
EXP="A5-2_resencl_plainz_topk_417_f0"
export TOPANEU_ROOT
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
log() { echo "[gpu1-c1 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$WORK"

n_nii() { ls "$1"/*.nii.gz 2>/dev/null | wc -l; }
n_ids() { "$ENVBIN/python" -c "
import json,os;print(len(json.load(open(os.environ['TOPANEU_ROOT']+'/dataset/TopAneu/dataset_split.json'))['splits']['$1']))"; }

# ---------------------------------------------------------------- STEP1 학습
if grep -q "finished with status 0" "$E/$EXP/train.log" 2>/dev/null; then
  log "STEP1 이미 완료 — 건너뜀"
else
  CONT=""
  if ls "$E/$EXP"/results/*/*/fold_0/checkpoint_latest.pth >/dev/null 2>&1; then
    CONT="--c"; log "STEP1 재개: checkpoint_latest에서 --c"
  else log "STEP1 시작: $EXP (공식 split fold0)"; fi
  GPU=1 NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$SCRIPTS/run_experiment.sh" 720 3d_fullres 0 "$EXP" \
    -p nnUNetResEncUNetLPlans -tr nnUNetTrainerTverskyTopkCE $CONT >> "$LOG" 2>&1
  grep -q "finished with status 0" "$E/$EXP/train.log" 2>/dev/null || { log "ABORT: 동맥류 학습 실패"; exit 1; }
  DA=$(grep -h "Mean Validation Dice" "$E/$EXP"/results/*/*/fold_0/training_log_*.txt 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  log "STEP1 완료 Dice=${DA:-N/A}"
  "$ENVBIN/python" "$SCRIPTS/make_summary.py" "$E/$EXP" >> "$LOG" 2>&1
  "$ENVBIN/python" "$SCRIPTS/make_leaderboard.py" >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------- STEP2~4 추론/후처리
for SP in val test; do
  N=$(n_ids "$SP")
  "$ENVBIN/python" - "$WORK/in_$SP" "$SP" <<'PY' >> "$LOG" 2>&1
import json, os, sys
from pathlib import Path
out, key = Path(sys.argv[1]), sys.argv[2]
root = Path(os.environ["TOPANEU_ROOT"])
ids = json.load(open(root/"dataset"/"TopAneu"/"dataset_split.json"))["splits"][key]
img = root/"nnunet"/"nnUNet_raw"/"Dataset720_TopAneuBinary417"/"imagesTr"
out.mkdir(parents=True, exist_ok=True)
n = 0
for cid in ids:
    s, d = img/f"{cid}_0000.nii.gz", out/f"{cid}_0000.nii.gz"
    if s.exists() and not d.exists():
        os.symlink(s, d); n += 1
print(f"[prep] {key}: 신규 {n}개 링크 (총 {len(ids)})")
PY

  if [ "$(n_nii "$WORK/aneu_$SP")" -ge "$N" ]; then log "STEP2($SP) 이미 완료 — 건너뜀"; else
    log "STEP2($SP): 동맥류 추론 ($EXP, fold0)"
    nnUNet_results="$E/$EXP/results" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
      -i "$WORK/in_$SP" -o "$WORK/aneu_$SP" -d 720 -c 3d_fullres -f 0 \
      -tr nnUNetTrainerTverskyTopkCE -p nnUNetResEncUNetLPlans -chk checkpoint_best.pth \
      --disable_tta -npp 2 -nps 2 --continue_prediction >> "$LOG" 2>&1
    log "STEP2($SP) 종료 ($(n_nii "$WORK/aneu_$SP")/$N)"
  fi

  if [ "$(n_nii "$WORK/ves_$SP")" -ge "$N" ]; then log "STEP3($SP) 이미 완료 — 건너뜀"; else
    log "STEP3($SP): 혈관 추론 (V4-2, fold0)"
    nnUNet_results="$E/V4-2_vessel_classweighted_417_500ep/results" CUDA_VISIBLE_DEVICES=1 "$ENVBIN2/nnUNetv2_predict" \
      -i "$WORK/in_$SP" -o "$WORK/ves_$SP" -d 800 -c 3d_fullres -f 0 \
      -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep -p nnUNetResEncUNetMPlans \
      -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 --continue_prediction >> "$LOG" 2>&1
    log "STEP3($SP) 종료 ($(n_nii "$WORK/ves_$SP")/$N)"
  fi

  if [ "$(n_nii "$WORK/vespp_$SP")" -ge "$N" ]; then log "STEP4($SP) 이미 완료 — 건너뜀"; else
    log "STEP4($SP): 혈관 후처리"
    "$ENVBIN/python" "$SCRIPTS/postprocess_vessel.py" apply "$WORK/ves_$SP" "$WORK/vespp_$SP" >> "$LOG" 2>&1
  fi
done

# ---------------------------------------------------------------- STEP5 C1/C2 평가
cd "$SCRIPTS"
for SP in val test; do
  for PAIR in "vespp_$SP:pp" "ves_$SP:raw"; do
    D="${PAIR%%:*}"; T="${PAIR##*:}"
    log "STEP5: C1 $SP / vessel=$T"
    "$ENVBIN/python" d900_infer_eval.py --split "$SP" \
      --aneurysm-pred-dir "$WORK/aneu_$SP" --vessel-pred-dir "$WORK/$D" \
      --lookup "$ANALYSIS/d9xx_lookup_V2_nearest_r2.json" --tag "realpred_${T}" >> "$LOG" 2>&1
    log "STEP5: C2 $SP / vessel=$T"
    "$ENVBIN/python" d910_infer_eval.py --split "$SP" \
      --aneurysm-pred-dir "$WORK/aneu_$SP" --vessel-pred-dir "$WORK/$D" \
      --index "$ANALYSIS/d910_index_k5_p1.npz" --tag "realpred_${T}" >> "$LOG" 2>&1
  done
done
log "GPU1 체인 완료 — analysis/d9xx_eval_*_realpred_*.json, d910_eval_*_realpred_*.json"
