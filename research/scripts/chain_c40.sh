#!/usr/bin/env bash
# C40 — ET 기준 분류 천장 완성 (2026-08-18).
# C39 에서 test 천장(예측 혈관)만 쟀다. 남은 칸을 채워 "분류가 어디까지 갈 수 있나"를 확정한다.
#   1) val 천장 (예측 혈관)     — split 간 일관성 확인
#   2) test 천장 (참조 혈관마스크) — 혈관을 정답으로 바꾸면 얼마나 더 오르나
# 2)가 중요한 이유: 앞서 RF 로 쟀을 때 참조혈관과 예측혈관의 천장 차이가 ±0.02 노이즈였다.
# ET 로 분류가 세진 지금도 같은지 확인해야 '혈관 개선 무용' 결론이 유지된다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
log(){ echo "[c40 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ceil(){ # $1=split $2=tag $3=vessel-dir $4=bp-dir
  [ -f "$A/c5_eval_${1}_${2}.json" ] && return
  log "  $1 $2  (혈관=$(basename $3))"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$1" \
    --vessel-dir "$3" --bp-dir "$4" \
    --model et --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$2" 2>&1 | grep -E '"MCC"|"top1_accuracy"' | head -2 || log "    실패"
}

log "=== ET 분류 천장 ==="
ceil val  c40_ceil_val_pp    "$P/vespp_val"  "$BP/val_pred"
ceil test c40_ceil_test_ref  "$REFVES"       "$BP/all_ref"

log "=== 요약 (천장, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_*_c39_ceil*.json"))+
                glob.glob(os.path.join(sys.argv[1],"c5_eval_*_c40_ceil*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c36_et.json"),
                 os.path.join(sys.argv[1],"c5_eval_val_c36_et.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    sp="test" if "_test_" in os.path.basename(f) else "val"
    ves=os.path.basename(str(d.get("vessel_dir","")))
    kind="천장" if "GT" in str(d.get("aneurysm_pred_dir","")) else "e2e"
    rows.append((f"{os.path.basename(f)[len('c5_eval_'):-5]}",sp,kind,ves,o,c,d.get("top1_accuracy")))
print(f"\n{'설정':<26}{'split':<6}{'구분':<6}{'혈관':<12}{'off.MCC':>9}{'cov.MCC':>9}{'cov.복합':>9}{'top1':>8}")
for t,sp,k,v,o,c,t1 in sorted(rows,key=lambda r:(r[1],-r[5]['MCC'])):
    print(f"{t:<26}{sp:<6}{k:<6}{v:<12}{o['MCC']:>9.4f}{c['MCC']:>9.4f}{comp(c):>9.4f}{(t1 or 0):>8.3f}")
PYEOF
log "=== 완료 ==="
