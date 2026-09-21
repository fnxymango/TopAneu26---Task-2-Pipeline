#!/usr/bin/env bash
# E-0 러너. 규칙은 e0_icaexpert.py 머리말에 고정.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee; D=$R/experiments/D1_newdata
SC=/tmp/scratch
export TOPANEU_ROOT=$R OMP_NUM_THREADS=6
export PYTHONPATH=$R/code/sblee/nnunet/scripts:$R/code/sblee:$R/code/sblee/nnunet:$SC
exec 9>"$D/e0.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][e0] $*" | tee -a "$D/STATUS.log"; }
log "ICA 전용분류기(질량보존 재분배) · LOCO 214폴드 · 시드 0,1,2"
"$HOME/miniconda3/envs/sbaneu2/bin/python" -u "$D/e0_icaexpert.py" > "$D/E0.out" 2>&1
rc=$?; touch "$D/.done_e0"
log "E0 끝(rc=$rc) → $(grep -o '\*\*판정 →.*' $R/experiments/V1_vessel_axis/RESULTS_E0.md 2>/dev/null | tail -1)"
