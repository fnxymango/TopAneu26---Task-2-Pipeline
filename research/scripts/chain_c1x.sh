#!/usr/bin/env bash
# C11~C13 방법론 실험 큐 (2026-08-15). 각 실험은 독립이며 순차 실행한다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; A="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
log(){ echo "[c1x $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
FEAT="$A/c10_feat_train.json"

log "=== C13 좌우 canonical화 ==="
$PY -u c13_lat_canonical.py --feat "$FEAT" --folds 5
log "=== C12 그룹 조건부 전문가 ==="
$PY -u c12_group_experts.py --feat "$FEAT" --folds 5
log "=== C11 희귀클래스 합성 ==="
if [ -f "$S/c11_synth_rare.py" ]; then
  $PY -u c11_synth_rare.py --feat "$FEAT" --bp-dir "$R/experiments/_c4_bpgraph/all_ref" --folds 5
else
  log "C11 스크립트 아직 없음 — 건너뜀"
fi
log "=== 큐 완료 ==="
