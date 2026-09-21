#!/usr/bin/env bash
# C11 / C15 / C14 실험 큐 (2026-08-15). 기대이득·리스크 순으로 배치.
#   C11 희귀클래스 합성  — AChA 0% 문제 직격, 기대이득 최대
#   C15 측지거리 좌표    — 유클리드 대신 혈관따라 거리, C10과 상보 기대
#   C14 혈관망 내부표현  — 가장 무겁고 실패 위험 큼(CPU 전용, GPU 학습 보호)
# 앞 실험이 실패해도 뒤가 계속 돌도록 각 단계를 독립 실행한다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; A="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
BP="$R/experiments/_c4_bpgraph/all_ref"
log(){ echo "[c11-15 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# chain_best 가 아직 돌면 CPU 경합을 피해 대기
while pgrep -f chain_best.sh >/dev/null; do log "chain_best 대기중..."; sleep 120; done

log "=== C11 희귀클래스 합성 ==="
$PY -u c11_synth_rare.py --feat "$FEAT" --bp-dir "$BP" --vessel-dir "$REFVES" \
    --folds 5 --max-cases 120 || log "C11 실패 status=$?"

log "=== C15 측지거리 좌표 ==="
$PY -u c15_geodesic_coords.py --feat "$FEAT" --vessel-dir "$REFVES" --bp-dir "$BP" \
    --folds 5 || log "C15 실패 status=$?"

log "=== C14 혈관망 내부표현 (CPU) ==="
$PY -u c14_vessel_encoder_feat.py --feat "$FEAT" --folds 5 --max-cases 60 \
    || log "C14 실패 status=$?"

log "=== 큐 완료 ==="
