#!/usr/bin/env bash
# B1BASE — 관문 V 기준 팔: 제출 혈관 모델로 val 41 재추론 → V5 후처리 → C4 그래프 (GPU0)
B=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/B1_vessel_contact_ft
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee nnUNet_raw=/tmp nnUNet_preprocessed=/tmp nnUNet_results=/tmp
exec 9>"$B/b1base.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b1base] $*" | tee -a "$D/STATUS.log"; }
log "기준 팔 val 재추론 시작 · GPU0"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$B/b1_val.py" predict base && "$PY" -u "$B/b1_val.py" post base && touch "$B/.done_b1base"
log "기준 팔 끝 · done=$([ -f $B/.done_b1base ] && echo y || echo n)"
