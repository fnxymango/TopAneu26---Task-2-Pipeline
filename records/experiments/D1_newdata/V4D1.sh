#!/usr/bin/env bash
# V4D1 — V4-D 1단계: train 검출 blob OOF 학습표 (v4d_oof.py · GPU 2장 · E9 fold6~9) · 이후 관문은 v4d_oof.py 머리말 고정
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee nnUNet_raw=/tmp nnUNet_preprocessed=/tmp nnUNet_results=/tmp
exec 9>"$D/v4d1.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v4d1] $*" | tee -a "$D/STATUS.log"; }
log "V4-D 1단계 시작 · train 291 OOF 검출 blob 학습표 · GPU0/GPU1"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$D/v4d_oof.py" run 0 2 > "$D/v4d1_g0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$D/v4d_oof.py" run 1 2 > "$D/v4d1_g1.log" 2>&1 &
wait
"$PY" "$D/v4d_oof.py" merge > "$D/../V1_vessel_axis/RESULTS_V4D1.md" 2>&1 && touch "$D/.done_v4d1"
log "V4-D 1단계 끝 → $(tail -2 $D/../V1_vessel_axis/RESULTS_V4D1.md | tr '\n' ' ')"
