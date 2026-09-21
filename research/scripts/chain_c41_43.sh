#!/usr/bin/env bash
# C41→C43 — 앵커 부재 문제를 순서대로 (2026-08-18). 세션과 무관하게 끝까지 돈다.
#
# 배경: 52클래스 중 61%가 분기점으로 정의되는데, 참조 마스크에서조차 Pcom/AChA/OA 계열
# 7종의 검출률이 41~60%다. 그 케이스들에서 3.4/3.5/3.6 은 **동일한 피처**를 받으므로
# 어떤 분류기도 구분할 수 없다. 오늘 모델·결정규칙·피처를 다 바꿔봐도 안 움직인 이유다.
#
#   C41  진단 — 앵커가 없는 게 분할 문제인가 추출 문제인가 (처방이 갈린다)
#   C42  결측 지시자 34차원 — 지금은 '멀다'와 '없다'가 둘 다 0 이라 구분이 안 된다
#   C43  아틀라스 대체값 34차원 — 랜드마크(98%+) 기준 기댓값 위치로 결측을 채운다
#
# 선택은 val, 보고는 test. 학습은 train 268 만 (PROJECT_RULES.md §1).
# 블록은 학습쪽 피처 파일에도 채워 넣어 학습·추론 대칭을 맞춘다(C25 가 걸린 함정).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"; ATLAS="$A/c42_anchor_atlas.json"; FEAT2="$A/c42_feat_train_anchor.json"
log(){ echo "[c4x $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── C41: 진단 ─────────────────────────────────────────────────────────
if [ ! -f "$A/c41_anchor_diag.json" ]; then
  log "=== C41: 앵커 부재 원인 진단 (분할 vs 추출) ==="
  $PY -u c41_anchor_diag.py 2>&1 | tail -45 || log "  실패"
else
  log "C41 이미 완료"
fi

# ── C42/C43 준비: 아틀라스 + 학습 피처 채우기 ─────────────────────────
log "=== C42: 아틀라스 구축 (train 292 에서만) ==="
[ -f "$ATLAS" ] || $PY -u c42_anchor_atlas.py build --bp-dir "$BP/all_ref" --split train \
    --out "$ATLAS" 2>&1 | tail -10 || { log "  아틀라스 실패"; exit 1; }
log "=== C42: 학습 피처에 블록 채우기 ==="
[ -f "$FEAT2" ] || $PY -u c42_anchor_atlas.py fillfeat --feat "$FEAT" --bp-dir "$BP/all_ref" \
    --atlas "$ATLAS" --out "$FEAT2" 2>&1 | tail -4 || { log "  채우기 실패"; exit 1; }

ev(){ # $1=split $2=tag $3=flags
  local sp=$1 tag=$2 ex=$3 VD VB AD
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; AD="$P/aneu_val_probavgf"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag  [$ex]"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT2" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" --atlas "$ATLAS" \
    --model et --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 $ex \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

log "=== C42/C43: val 비교 ==="
ev val c4x_base   ""
ev val c4x_miss   "--use-miss"
ev val c4x_imp    "--use-imp"
ev val c4x_both   "--use-miss --use-imp"

BEST=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c4x_*.json")):
    o=json.load(open(f)).get("official_div52") or {}
    n=os.path.basename(f)[len("c5_eval_val_"):-5]
    if o: rows.append((o.get("MCC",-1),n))
rows.sort(reverse=True)
for v,n in rows: print(f"  {n:<14} {v:.4f}", file=sys.stderr)
print(rows[0][1] if rows else "c4x_base")
PYEOF
)
log "=== val 최고: $BEST ==="

# val 이득이 0.03 미만이면 노이즈로 본다 — C39 에서 val +0.006 을 따라갔다가 test 에서 -0.074 였다
GAIN=$($PY - "$A" "$BEST" <<'PYEOF'
import json,os,sys
A,b=sys.argv[1],sys.argv[2]
g=lambda t:(json.load(open(os.path.join(A,f"c5_eval_val_{t}.json"))).get("official_div52") or {}).get("MCC",0)
print(f"{g(b)-g('c4x_base'):.4f}")
PYEOF
)
log "=== 기준선 대비 val 이득 $GAIN (0.03 미만이면 노이즈) ==="

log "=== test 확인 ==="
ev test c4x_base ""
case "$BEST" in
  c4x_miss) ev test c4x_miss "--use-miss" ;;
  c4x_imp)  ev test c4x_imp  "--use-imp" ;;
  c4x_both) ev test c4x_both "--use-miss --use-imp" ;;
esac

log "=== 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_c4x_*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c36_et.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5],o,c))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<16}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}"
      f"{'cov.Dice':>9}{'cov.HD95':>9}{'cov.복합':>9}")
for tag,o,c in rows:
    print(f"{tag:<16}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}"
          f"{c['MCC']:>8.4f}{c['DICE']:>9.4f}{c['HD95']:>9.4f}{comp(c):>9.4f}")
PYEOF
log "=== 완료 ==="
