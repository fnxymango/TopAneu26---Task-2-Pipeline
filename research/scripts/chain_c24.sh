#!/usr/bin/env bash
# C24 — 크롭 분류기 전체 큐 (2026-08-17). 중간에 끊기지 않게 **한 스크립트**로 묶는다.
#   a) 크롭 생성 (train / val GT / test GT / val·test 검출)
#   b) 게이트1(좌표) → 게이트2(영상신호) → 학습 → 기하RF 앙상블
# 대기 조건에 pgrep 를 절대 쓰지 않는다(앞서 두 번 정체시킨 원인). 산출물 개수로 판정.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; CR="$E/_c24_crops"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
FEAT="$A/c10_feat_train.json"
log(){ echo "[c24 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

mkcrop(){ # $1=split $2=vessel $3=out $4=aneu(옵션)
  local n; n=$(ls "$3"/*.npz 2>/dev/null | wc -l)
  [ "$n" -gt 10 ] && { log "  $3 이미 있음 ($n)"; return; }
  if [ -n "${4:-}" ]; then
    $PY -u c24_build_crops.py --split "$1" --vessel-dir "$2" --out "$3" --aneu-dir "$4" 2>&1 | tail -3
  else
    $PY -u c24_build_crops.py --split "$1" --vessel-dir "$2" --out "$3" 2>&1 | tail -3
  fi
}

log "=== STEP1: 크롭 생성 ==="
mkcrop train "$REFVES"          "$CR/train"
mkcrop val   "$P/vespp_val"     "$CR/val_gt"
mkcrop test  "$P/vespp_test"    "$CR/test_gt"
mkcrop val   "$P/vespp_val"     "$CR/val_det"  "$P/aneu_val_probavgf"
mkcrop test  "$P/vespp_test"    "$CR/test_det" "$P/aneu_test_probavgf"
log "STEP1 완료 — train $(ls $CR/train/*.npz 2>/dev/null|wc -l) · val_gt $(ls $CR/val_gt/*.npz 2>/dev/null|wc -l) · test_gt $(ls $CR/test_gt/*.npz 2>/dev/null|wc -l)"

log "=== STEP2: 게이트 → 학습 → 앙상블 ==="
$PY -u c24_train_eval.py --crop-dir "$CR/train" --feat "$FEAT" --epochs 120 2>&1 || log "STEP2 실패"

log "=== 완료 ==="
