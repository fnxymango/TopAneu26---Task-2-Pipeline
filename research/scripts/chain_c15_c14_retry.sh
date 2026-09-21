#!/usr/bin/env bash
# C15/C14 재시도 (2026-08-15). 실패 원인: 피처파일에 _cen(병변 중심)이 없었음.
# _cen 은 c5.extract_case_rows 에 나중에 추가돼서, 그 전에 빌드된 c10_feat_train.json 에는 없다.
# -> 최신 c5 로 train 피처를 다시 빌드(pos + _cen 동시 포함)한 뒤 두 실험을 다시 돌린다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; A="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
BP="$R/experiments/_c4_bpgraph/all_ref"
FEAT="$A/c10_feat_train.json"
log(){ echo "[retry $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

if ! $PY -c "import json,sys; d=json.load(open('$FEAT')); sys.exit(0 if d and d[0].get('_cen') else 1)" 2>/dev/null; then
  log "피처 재빌드 (pos + _cen 포함)"
  cp "$FEAT" "$FEAT.bak" 2>/dev/null
  $PY -u c5_location_v2.py build --split train --vessel-dir "$REFVES" --bp-dir "$BP" --out "$FEAT"
  log "재빌드 종료 status=$?"
fi

log "=== C15 측지거리 좌표 ==="
$PY -u c15_geodesic_coords.py --feat "$FEAT" --vessel-dir "$REFVES" --bp-dir "$BP" --folds 5 \
  || log "C15 실패 status=$?"

log "=== C14 혈관망 내부표현 (CPU) ==="
$PY -u c14_vessel_encoder_feat.py --feat "$FEAT" --folds 5 --max-cases 60 \
  || log "C14 실패 status=$?"

log "=== 재시도 큐 완료 ==="
