#!/usr/bin/env bash
# E14 — 추론 TTA(좌우 미러 평균) 를 test 시드 5판으로 처음 제대로 잰다 (2026-08-19).
#
# 왜 지금: 오늘 후보를 전수 정리해보니 TTA 만 **e2e test 를 한 번도 안 재봤다**.
#   E5 반복 CV     +0.0082 (t=1.8)   — 애매
#   val 단발       0.4531 (기준 0.4632) — 애매
# 그 상태로 덮였다. 그런데 TTA 는 **차원을 안 늘리고 추론만 바꾸는** 방식이라
# 오늘 살아남은 것들(β 교정·τ 게이트)과 같은 부류이고, 진 것들(측지·호위치·앵커·크롭 —
# 전부 차원 추가)과 다른 부류다. 제대로 재볼 값이 있다.
#
# 방식: 추론 시 원본과 좌우 미러본을 둘 다 예측해 확률을 평균한다(미러 클래스명 되돌림 포함).
# 학습은 그대로다. 비용은 추론 2배(건당 ~11분).
#
# E1b-2 가 우선이라 nice 19 로 돌린다 — 경합이 생기면 이쪽이 양보한다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[e14 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== TTA test 시드 5판 (기준선 e4_rf 와 짝지어 비교) ==="
for SD in 0 1 2 3 4; do
  TAG="e14_tta_s${SD}"
  [ -f "$A/c5_eval_test_${TAG}.json" ] && { log "  $TAG 있음"; continue; }
  log "  $TAG"
  CLF_SEED=$SD nice -n 19 $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split test \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
    --aneurysm-pred-dir "$P/aneu_test_probavgf" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 --tta \
    --tag "$TAG" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
done

log "=== 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]; K=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_test_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1))
        o[sd]=json.load(open(f))["adjusted_div_present"]
    return o
g,b=load("e14_tta"),load("e4_rf")
sds=sorted(set(g)&set(b))
if len(sds)<2: print(f"데이터 부족 (n={len(sds)})"); raise SystemExit
print(f"\n{'시드':>4}{'TTA':>10}{'기준선':>10}{'차이':>10}")
for s in sds: print(f"{s:>4}{g[s]['MCC']:>10.4f}{b[s]['MCC']:>10.4f}{g[s]['MCC']-b[s]['MCC']:>+10.4f}")
print(f"\n{'지표':<10}{'TTA':>18}{'기준선':>18}{'짝지은 차이':>16}{'유리시드':>9}")
for k in K:
    ga=np.array([g[s][k] for s in sds]); ba=np.array([b[s][k] for s in sds]); d=ga-ba
    w=int((d>0).sum()) if k!='HD95' else int((d<0).sum())
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0
    print(f"{k:<10}{ga.mean():>10.4f}±{ga.std():<7.4f}{ba.mean():>10.4f}±{ba.std():<7.4f}"
          f"{d.mean():>+9.4f}(t{t:+.1f}){w:>6}/{len(sds)}")
ga=np.array([g[s]['MCC'] for s in sds]); ba=np.array([b[s]['MCC'] for s in sds]); d=ga-ba
t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
print(f"\n  짝지은 차이 {d.mean():+.4f} ± {d.std(ddof=1):.4f}  t={t:+.2f}  이긴 시드 {int((d>0).sum())}/{len(d)}")
print(f"  [판정] {'★ 채택 후보 — val/292 로 확인 필요' if d.mean()>0 and t>2.5 else ('열세' if d.mean()<0 and t<-2.5 else '판정 불가 — 기준선 유지')}")
PYEOF
log "=== 완료 ==="
