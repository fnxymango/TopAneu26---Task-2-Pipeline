#!/usr/bin/env bash
# TAB0 — TabICLv2 OOF 관문 러너. 규칙은 tab0_oof.py 머리말에 고정.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata
SC=/tmp/scratch
export HF_HOME=$SC/hf TOPANEU_ROOT=$R OMP_NUM_THREADS=8
export PYTHONPATH=$R/code/sblee/nnunet/scripts:$R/code/sblee:$R/code/sblee/nnunet:$SC
exec 9>"$D/tab0.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][tab0] $*" | tee -a "$D/STATUS.log"; }
log "TabICLv2 대 RF · LOCO 214폴드 · 시드 0,1,2 · 관문(탈락 기준)"
"$SC/tabenv/bin/python" -u "$D/tab0_oof.py" > "$D/TAB0.out" 2>&1
rc=$?
touch "$D/.done_tab0"
log "TAB0 끝(rc=$rc) → $(grep -o '\*\*판정 →.*' $R/experiments/V1_vessel_axis/RESULTS_TAB0.md 2>/dev/null | tail -1)"
