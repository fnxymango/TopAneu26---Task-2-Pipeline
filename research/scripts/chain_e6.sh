#!/usr/bin/env bash
# E6 실행체인 — E4(시드 산포)가 끝나면 케이스 부트스트랩으로 **신뢰구간**을 붙인다.
#
# 두 종류의 불확실성을 분리해서 본다:
#   E4  분류기 시드만 바꿨을 때의 변동      -> "같은 데이터에서 다시 돌리면"
#   E6  test 83케이스를 재표집했을 때의 변동 -> "다른 83명이었다면"
# 개선폭을 채택하려면 **둘 다**보다 커야 한다. 지금까지는 둘 다 잰 적이 없다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[e6 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== E4 완료 대기 (평가 10건) ==="
for i in $(seq 1 60); do
  n=$(ls "$A"/c5_eval_test_e4_*.json 2>/dev/null | wc -l)
  [ "$n" -ge 10 ] && { log "  E4 완료 ($n/10)"; break; }
  log "  대기 $((i*5))분 — E4 $n/10"; sleep 300
done

# 케이스별 원자료가 없는 평가는 다시 돌린다 (per-case 저장 패치 이전에 돈 것들)
log "=== per-case 원자료 보충 ==="
for M in rf et; do
  for SD in 0 1 2 3 4; do
    TAG="e4_${M}_s${SD}"
    [ -f "$A/c5_percase_test_${TAG}.json" ] && continue
    [ -f "$A/c5_eval_test_${TAG}.json" ] || continue
    log "  재실행 $TAG (원자료 없음)"
    rm -f "$A/c5_eval_test_${TAG}.json"
    CLF_SEED=$SD $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split test \
      --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
      --aneurysm-pred-dir "$P/aneu_test_probavgf" \
      --model "$M" --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "$TAG" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
  done
done

log "=== 부트스트랩 2000회 ==="
$PY -u e6_bootstrap.py --n 2000 --split test 2>&1 | tail -40
log "=== 완료 ==="
