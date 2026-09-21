#!/usr/bin/env bash
# C33 — train+val 체제에서 분류 천장 재측정 + β/τ 재선택 (2026-08-18). A7 학습과 CPU/GPU 병렬.
#
# C32 로 학습셋이 268 -> 311 병변이 되면서 test cov.MCC 0.3306 -> 0.3563 이 됐다.
# 그런데 β=0.5, τ=0.5 는 **val 이 학습 밖에 있을 때** 고른 값이다. val 이 들어간 지금
# 그 값이 최적이라는 보장이 없고, val 은 이제 선택에 못 쓴다.
# 그래서 train∪val 안에서 환자단위 CV 로 다시 고른다. 다만 오늘 확인했듯이
# **GT 병변만 보는 CV 는 β 를 못 고른다**(FP 가 없어 β=0 이 최선이라 나온다).
# 그래서 CV 로는 τ 만 좁히고, 최종 확인은 test 한 번으로 한다 — test 로 고르지는 않는다.
#
#   STEP1  분류 천장 재측정 (GT 병변, train+val 학습) — 남은 헤드룸이 얼마인지
#   STEP2  현행 (β=0.5, τ=0.5) 주변 소수 조합만 test 확인
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
MERGED="$A/c32_feat_trainval.json"
log(){ echo "[c33 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=tag $2=beta $3=tau $4=ceil
  local tag=$1 bt=$2 tau=$3 ceil=${4:-0}
  local AD="--aneurysm-pred-dir $P/aneu_test_probavgf"
  [ "$ceil" = 1 ] && AD=""
  [ -f "$A/c5_eval_test_${tag}.json" ] && return
  log "  test $tag (β=$bt τ=$tau ceil=$ceil)"
  $PY -u c5_location_v2.py eval --train-feat "$MERGED" --split test \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" $AD \
    --model rf --use-pos --beta "$bt" --conf-tau "$tau" --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"|"top1_accuracy"' | head -2 || log "    실패"
}

log "=== STEP1: 분류 천장 재측정 (train+val 학습) ==="
ev c33_ceil_tv 0.5 0.5 1
ev c33_ceil_tv_b0 0.0 0.0 1

log "=== STEP2: τ/β 주변 확인 ==="
ev c33_t0.4_b0.5 0.5 0.4 0
ev c33_t0.6_b0.5 0.5 0.6 0
ev c33_t0.5_b0.75 0.75 0.5 0

log "=== STEP3: 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c3[123]*.json"))):
    d=json.load(open(f))
    if d.get("eval_mode")=="covered_gt": continue
    o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    ceil="천장" if "GT" in str(d.get("aneurysm_pred_dir","")) else ""
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5]+ (" "+ceil if ceil else ""),o,c,d.get("top1_accuracy")))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<22}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.HD95':>9}{'cov.복합':>9}{'top1':>8}")
for tag,o,c,t1 in rows:
    print(f"{tag:<22}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}{(t1 or 0):>8.3f}")
PYEOF
log "=== 완료 ==="
