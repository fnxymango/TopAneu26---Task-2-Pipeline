#!/usr/bin/env bash
# C38 — 확신 게이트가 실제 효과인지 노이즈인지 (2026-08-18).
# val 격자에서 τ 를 0.1 움직이면 MCC 가 0.03~0.04 흔들린다(42케이스·33병변).
# τ=0.5 의 우위(0.3040 vs 이웃 0.2901/0.2609)가 그 노이즈 안에 들어간다.
# test 에서 게이트 유무를 나란히 놓고 본다 — **고르는 게 아니라** 견고성 확인이다.
# 비슷하면 더 단순한 τ=0 을 채택한다(파라미터가 하나 줄고 val 과적합 위험도 준다).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[c38 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ local tag=$1 tau=$2
  [ -f "$A/c5_eval_test_${tag}.json" ] && return
  log "  test $tag (ET β=0.5 τ=$tau)"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split test \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
    --aneurysm-pred-dir "$P/aneu_test_probavgf" \
    --model et --use-pos --beta 0.5 --conf-tau "$tau" --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}
log "=== ET, 게이트 유무 test 비교 ==="
ev c38_et_nogate 0.0
ev c38_et_t0.6   0.6

log "=== 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c3[68]*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c31_t0.5_b0.5.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5],o,c))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<20}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.HD95':>9}{'cov.복합':>9}")
for tag,o,c in rows:
    print(f"{tag:<20}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}")
PYEOF
log "=== 완료 ==="
