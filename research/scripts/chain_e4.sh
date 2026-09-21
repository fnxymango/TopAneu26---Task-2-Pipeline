#!/usr/bin/env bash
# E4 — "그 MCC 가 시드를 바꿔도 재현되는가" (2026-08-19). 사용자 질문에 직접 답하는 실험.
#
# E2 에서 밝혀진 것: test 에서 RF 와 ET 의 정답 개수가 44/63 로 **똑같다**(3승3패).
# 그런데 cov.MCC 는 ET 가 +0.0303 높다. 차이가 전부 "어느 클래스를 맞혔나"에서 나왔다.
# ET 가 이긴 셋은 train 표본 5·5·9 인 희소 클래스, RF 가 이긴 셋은 7·10·32 인 흔한 클래스다.
#
# 그렇다면 선행 질문은 이것이다 — 모델을 안 바꾸고 **난수 시드만 바꿔도** 점수가
# 그만큼 움직이는가. ExtraTrees 는 분할점을 무작위로 뽑으므로 시드가 결과를 흔든다.
# 시드 산포가 ET-RF 격차(0.0303)와 같은 크기면, 그 격차는 모델 차이가 아니라 추첨이다.
#
# rf/et 각각 시드 0~4 로 test e2e 를 돌려 cov.MCC 의 평균과 산포를 낸다.
# 보고는 PROJECT_RULES.md 6-1 대로 official_div52 + covered_gt 둘 다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[e4 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

for M in rf et; do
  for SD in 0 1 2 3 4; do
    TAG="e4_${M}_s${SD}"
    [ -f "$A/c5_eval_test_${TAG}.json" ] && { log "  $TAG 이미 있음"; continue; }
    log "  test $M seed=$SD"
    CLF_SEED=$SD $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split test \
      --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
      --aneurysm-pred-dir "$P/aneu_test_probavgf" \
      --model "$M" --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "$TAG" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
  done
done

log "=== E4 결과: 시드 산포 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,numpy as np
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
acc={}
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_e4_*.json"))):
    b=os.path.basename(f); m=b.split("_")[3]; sd=int(b.split("_s")[1][0])
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    acc.setdefault(m,[]).append((sd,o['MCC'],comp(o),c['MCC'],comp(c),d.get('top1_accuracy')))
print(f"\n{'모델':<5}{'시드':>5}{'off.MCC':>10}{'off.복합':>10}{'cov.MCC':>10}{'cov.복합':>10}")
for m in ('rf','et'):
    for sd,om,oc,cm,cc,t1 in sorted(acc.get(m,[])):
        print(f"{m:<5}{sd:>5}{om:>10.4f}{oc:>10.4f}{cm:>10.4f}{cc:>10.4f}")
print()
stat={}
for m in ('rf','et'):
    if not acc.get(m): continue
    cm=np.array([r[3] for r in acc[m]]); om=np.array([r[1] for r in acc[m]])
    stat[m]=(cm,om)
    print(f"  {m}: cov.MCC {cm.mean():.4f} ± {cm.std():.4f}  (범위 {cm.min():.4f}~{cm.max():.4f}, 폭 {cm.ptp():.4f})")
    print(f"      off.MCC {om.mean():.4f} ± {om.std():.4f}  (폭 {om.ptp():.4f})")
if len(stat)==2:
    (ce,oe),(cr,orr)=stat['et'],stat['rf']
    gap=ce.mean()-cr.mean(); pooled=np.sqrt((ce.std()**2+cr.std()**2)/2)
    print(f"\n  [핵심] 보고된 ET-RF 격차 = +0.0303 (시드0 단발)")
    print(f"         시드평균 격차   = {gap:+.4f}")
    print(f"         시드 산포(합동) = {pooled:.4f}")
    print(f"  [판정] {'격차가 시드 노이즈보다 큼 — 재현됨' if gap > 2*pooled/np.sqrt(len(ce)) else '격차가 시드 산포에 묻힘 — 재현 안 됨'}")
PYEOF
log "=== 완료 ==="
