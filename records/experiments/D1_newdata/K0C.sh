#!/usr/bin/env bash
# K0C — 기준선 시드 5~9 생성(K0NULL) 이 끝나면 K0 판정 장치의 영가설 보정을 돌린다 (관문은 k0_judge.py 머리말 고정)
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][k0c] $*" | tee -a "$D/STATUS.log"; }
until [ -e "$D/.done_k0null" ]; do sleep 60; done
log "영가설 보정 시작 (기준선 10시드 · 5·5 분할 252 비교)"
"$PY" -u "$D/k0_judge.py" null > "$D/../V1_vessel_axis/RESULTS_K0_NULL.md" 2>&1 || { log "★보정 실패"; exit 1; }
log "영가설 보정 완료 → $(grep -o '보정 관문[^*]*' $D/../V1_vessel_axis/RESULTS_K0_NULL.md)"; touch "$D/.done_k0c"
