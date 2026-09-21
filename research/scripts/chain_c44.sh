#!/usr/bin/env bash
# C44 — 분기점 노드 반경 완화 (2026-08-18). C41 진단이 직접 지시한 처방.
#
# C41 이 원인을 갈랐다. 검출률 60% 미만 7종에서 분할손실 37.4% · 추출손실 16.6% 인데,
# **종류별로 보면 완전히 다르다**:
#   R/L-OA junction   혈관존재 97.6/98.8%  노드검출 59.7/65.2%  -> 추출손실 37.9/33.6% (거의 순수 추출)
#   R/L-A1A2+A3       혈관존재 97.8/96.4%  노드검출 71.5/76.5%  -> 추출손실 26.4/19.9%
#   R/L-Pcom          혈관존재 52~53%                          -> 분할 문제. 우회 외 방법 없음
# 즉 3.2(OA junction) 와 4.3/4.4(A2·A3 경계)는 **혈관이 있는데 노드가 안 생기는 것**이고,
# 노드 형성 반경(JUNCTION_R_MM=2.0)이 빡빡해서다. 재학습 없이 파라미터로 회수된다.
#
# 이건 C42/C43 처럼 차원을 더하는 게 아니라 **이미 있는 34차원의 빈 칸을 채우는 것**이라,
# 오늘 모든 피처 추가가 실패한 패턴(268샘플에 차원 추가는 손해)에 해당하지 않는다.
#
# 학습(참조마스크)·추론(예측마스크) 양쪽을 같은 반경으로 다시 뽑아야 대칭이 맞는다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
log(){ echo "[c44 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

for JR in 3.5 5.0; do
  T=$(echo $JR | tr -d .)
  log "=== JUNCTION_R_MM=$JR : 그래프 재추출 ==="
  for SRC in "all_ref:$REFVES" "vespp_val:$P/vespp_val" "vespp_test:$P/vespp_test"; do
    NAME=${SRC%%:*}; DIR=${SRC#*:}
    OUT="$BP/${NAME}_jr$T"
    NEED=$(ls "$DIR"/*.nii.gz 2>/dev/null | wc -l)
    HAVE=$(ls "$OUT"/*.json 2>/dev/null | wc -l)
    if [ "$HAVE" -ge "$((NEED-3))" ] && [ "$NEED" -gt 0 ]; then log "  $NAME 이미 있음 ($HAVE)"; continue; fi
    log "  $NAME ($NEED케이스)"
    $PY -u c4_branchpoint_graph.py --vessel-dir "$DIR" --out "$OUT" \
       --junction-r-mm "$JR" 2>&1 | tail -2 || log "    실패"
  done

  log "=== JR=$JR : 앵커 검출률 확인 ==="
  $PY - "$BP/all_ref_jr$T" <<'PYEOF'
import json,glob,sys,collections
from pathlib import Path
import c5_location_v2 as C5
cnt=collections.Counter(); tot=0
for f in glob.glob(str(Path(sys.argv[1])/"*.json")):
    tot+=1; seen=set()
    for nd in json.load(open(f)).get("nodes",[]):
        cs=set(nd["classes"])
        for i,p in enumerate(C5.JUNCTION_PAIRS):
            if set(p)<=cs: seen.add(i)
    for i in seen: cnt[i]+=1
watch=["R-ICA-C6-C7 + R-OA","L-ICA-C6-C7 + L-OA","R-A1A2 + R-A3","L-A1A2 + L-A3",
       "R-ICA-C6-C7 + R-AChA","L-ICA-C6-C7 + L-AChA","R-ICA-C6-C7 + R-Pcom"]
base={"R-ICA-C6-C7 + R-OA":59.7,"L-ICA-C6-C7 + L-OA":65.2,"R-A1A2 + R-A3":71.5,
      "L-A1A2 + L-A3":76.5,"R-ICA-C6-C7 + R-AChA":43.2,"L-ICA-C6-C7 + L-AChA":46.3,
      "R-ICA-C6-C7 + R-Pcom":44.8}
print(f"  {tot}케이스 · 관심 앵커 검출률 (JR=2.0 대비)")
for i,(a,b) in enumerate(C5.JUNCTION_PAIRS):
    k=f"{a} + {b}"
    if k in watch:
        r=cnt[i]/tot*100
        print(f"    {k:<30} {r:>5.1f}%  (기존 {base[k]:>5.1f}%, {r-base[k]:+.1f}p)")
PYEOF
done

log "=== 피처 재추출 + 평가 ==="
ev(){ local sp=$1 tag=$2 tf=$3 bpd=$4 VD AD
  if [ "$sp" = val ]; then VD=vespp_val; AD="$P/aneu_val_probavgf"; else VD=vespp_test; AD="$P/aneu_test_probavgf"; fi
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag"
  $PY -u c5_location_v2.py eval --train-feat "$tf" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$bpd" --aneurysm-pred-dir "$AD" \
    --model et --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}
for JR in 3.5 5.0; do
  T=$(echo $JR | tr -d .)
  F="$A/c44_feat_train_jr$T.json"
  if [ ! -f "$F" ]; then
    log "  train 피처 재추출 (JR=$JR)"
    $PY -u c5_location_v2.py build --split train --vessel-dir "$REFVES" \
       --bp-dir "$BP/all_ref_jr$T" --out "$F" 2>&1 | tail -2 || log "    실패"
  fi
  ev val "c44_jr$T" "$F" "$BP/vespp_val_jr$T"
done
ev val c44_base "$A/c10_feat_train.json" "$BP/val_pred"

BEST=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c44_*.json")):
    o=json.load(open(f)).get("official_div52") or {}
    n=os.path.basename(f)[len("c5_eval_val_"):-5]
    if o: rows.append((o.get("MCC",-1),n))
rows.sort(reverse=True)
for v,n in rows: print(f"  {n:<12} {v:.4f}",file=sys.stderr)
print(rows[0][1] if rows else "c44_base")
PYEOF
)
log "=== val 최고: $BEST ==="
case "$BEST" in
  c44_jr*) T=${BEST#c44_jr}; ev test "$BEST" "$A/c44_feat_train_jr$T.json" "$BP/vespp_test_jr$T" ;;
  *) log "  완화가 도움 안 됨 — test 생략" ;;
esac
log "=== 완료 ==="
