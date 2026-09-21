#!/usr/bin/env bash
# V5S — 새 피처 F1(돌출 방향)·F2(굵기)·F3(뼈) 추출이 끝나면 train OOF 스크리닝(v5_screen.py, 관문은 머리말 고정)
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v5s] $*" | tee -a "$D/STATUS.log"; }
until grep -q "^완료" "$D/v5_feat.out" 2>/dev/null && grep -q "^완료" "$D/v5_bone.out" 2>/dev/null; do
  grep -q Traceback "$D/v5_feat.out" "$D/v5_bone.out" 2>/dev/null && { log "★추출 실패 — Traceback"; exit 1; }
  sleep 30; done
log "추출 완료 · $(tail -1 $D/v5_feat.out) · $(tail -1 $D/v5_bone.out)"
log "train OOF 스크리닝 시작 (5팔 × 케이스5겹 × 시드5)"
"$PY" -u "$D/v5_screen.py" > "$D/v5_screen.out" 2>&1 || { log "★스크리닝 실패"; exit 1; }
cp "$D/v5_screen.out" "$D/../V1_vessel_axis/RESULTS_V5_SCREEN.md"
log "V5 스크리닝 완료 → $(tail -1 $D/v5_screen.out)"; touch "$D/.done_v5s"
