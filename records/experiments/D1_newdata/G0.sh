#!/usr/bin/env bash
# G-0 러너. 규칙은 g0_graph.py 머리말에 고정.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee; D=$R/experiments/D1_newdata
SC=/tmp/scratch
export TOPANEU_ROOT=$R OMP_NUM_THREADS=8
export PYTHONPATH=$R/code/sblee/nnunet/scripts:$R/code/sblee:$R/code/sblee/nnunet:$SC
exec 9>"$D/g0.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g0] $*" | tee -a "$D/STATUS.log"; }
log "연결관계 블록 3팔(기준/전체/비관계형) · LOCO 214폴드 · 시드 0,1,2"
"$HOME/miniconda3/envs/sbaneu2/bin/python" -u "$D/g0_graph.py" > "$D/G0.out" 2>&1
rc=$?; touch "$D/.done_g0"
log "G0 끝(rc=$rc) → $(grep -o '\*\*판정 →.*' $R/experiments/V1_vessel_axis/RESULTS_G0.md 2>/dev/null | tail -1)"
