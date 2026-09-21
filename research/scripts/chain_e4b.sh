#!/usr/bin/env bash
# E4b — test 에 보고된 **모든 구별되는 e2e 설정**에 시드 산포를 붙인다 (2026-08-19).
#
# 사용자 우선순위: "현재 e2e 모델들 성능확보가 최우선 — 재현되는지랑 재현되는 애들 중 최고".
# E4 는 현행 2종(rf/et, β0.5 τ0.5)만 5시드로 돌렸다. 나머지 보고 설정도 같은 대우를 해야
# "재현되는 애들 중 최고"를 고를 수 있다.
#
# test 보고 이력(cov.MCC, 중복 태그 제거):
#   0.3609  et β0.5 τ0.5   (c36_et = a8_base = c4x_base)   ← E4 가 담당
#   0.3597  et β0.5 τ0     (c38_et_nogate)
#   0.3563  rf β0.5 τ0.5 + val 학습 (c32_trainval)  ← **제외**: val 학습 금지(2026-08-18 지시)
#   0.3561  et β0.5 τ0.6   (c38_et_t0.6)
#   0.3306  rf β0.5 τ0.5   (c36_rf = c32_trainonly = c31_t0.5_b0.5)  ← E4 가 담당
#   0.3233  rf β0.5 τ0     (c26_b0.5 = c27_base)
#   0.3141  rf β1.0 τ0     (c26_b1.0)
#
# 여기서 돌리는 4종 x 5시드. 시드를 바깥 루프로 둬서 중간에 끊겨도 한 판은 갖춰지게 한다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[e4b $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== E4 완료 대기 ==="
for i in $(seq 1 60); do
  n=$(ls "$A"/c5_eval_test_e4_*.json 2>/dev/null | wc -l)
  [ "$n" -ge 10 ] && { log "  E4 완료 ($n/10)"; break; }
  log "  대기 $((i*5))분 — E4 $n/10"; sleep 300
done

ev(){ # $1=model $2=beta $3=tau $4=seed
  local tag="e4b_${1}_b${2}_t${3}_s${4}"
  [ -f "$A/c5_eval_test_$tag.json" ] && { log "  $tag 있음"; return; }
  log "  $tag"
  CLF_SEED=$4 $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split test \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
    --aneurysm-pred-dir "$P/aneu_test_probavgf" \
    --model "$1" --use-pos --beta "$2" --conf-tau "$3" --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

log "=== 4종 x 5시드 ==="
for SD in 0 1 2 3 4; do
  ev et 0.5 0.0 "$SD"
  ev et 0.5 0.6 "$SD"
  ev rf 0.5 0.0 "$SD"
  ev rf 1.0 0.0 "$SD"
done
log "=== 완료 ==="
