#!/usr/bin/env bash
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee; D=$R/experiments/D1_newdata
export TOPANEU_ROOT=$R OMP_NUM_THREADS=1
export PYTHONPATH=$R/code/sblee/nnunet/scripts:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/fpchar.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][fpchar] $*" | tee -a "$D/STATUS.log"; }
log "FP 해부 — 지표 단위(케이스×클래스)로 · 기준 b1fg_pf · 시드 0~4"
"$HOME/miniconda3/envs/sbaneu2/bin/python" -u "$D/fpchar.py" > "$D/FPCHAR.out" 2>&1
rc=$?; touch "$D/.done_fpchar"; log "FPCHAR 끝(rc=$rc)"
