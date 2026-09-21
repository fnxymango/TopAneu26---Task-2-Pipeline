#!/usr/bin/env bash
# K3P — K3 사전 점검(k3_pilot.py) · GPU 2장 병렬 · train ICA 원위 60케이스 · 관문은 k3_pilot.py 머리말 고정
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee nnUNet_raw=/tmp nnUNet_preprocessed=/tmp nnUNet_results=/tmp
exec 9>"$D/k3p.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][k3p] $*" | tee -a "$D/STATUS.log"; }
log "K3 사전 점검 시작 · 60케이스 · GPU0/GPU1"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$D/k3_pilot.py" run 0 2 > "$D/k3p_g0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PY" -u "$D/k3_pilot.py" run 1 2 > "$D/k3p_g1.log" 2>&1 &
wait
"$PY" "$D/k3_pilot.py" report > "$D/../V1_vessel_axis/RESULTS_K3_PILOT.md" 2>&1
log "K3 사전 점검 완료 → $(grep -o '관문[^*]*' $D/../V1_vessel_axis/RESULTS_K3_PILOT.md)"; touch "$D/.done_k3p"
