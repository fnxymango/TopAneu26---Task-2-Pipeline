#!/usr/bin/env bash
# V4D2 — V4-D 2단계 스크리닝(v4d_screen.py · train OOF 만). .done_v4d1 을 기다렸다가 실행. 관문은 v4d_oof.py 머리말 고정
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
exec 9>"$D/v4d2.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v4d2] $*" | tee -a "$D/STATUS.log"; }
log "대기 · .done_v4d1"
while [ ! -f "$D/.done_v4d1" ]; do sleep 60; done
log "V4-D 2단계 스크리닝 시작 · 3팔 × 5시드"
"$PY" "$D/v4d_screen.py" > "$D/../V1_vessel_axis/RESULTS_V4D_SCREEN.md" 2>"$D/v4d2.err" && touch "$D/.done_v4d2"
log "V4-D 2단계 끝 → $(grep -o '관문[^*]*' $D/../V1_vessel_axis/RESULTS_V4D_SCREEN.md)"
