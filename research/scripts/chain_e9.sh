#!/usr/bin/env bash
# E9 — geo(C15 측지거리) 를 시드 5판으로 확정한다 (2026-08-19). E1b 보다 우선.
#
# E7(시드 3판) 결과: geo 0.3716±0.0091 vs 현행 0.3477±0.0199, Δ +0.0240, t=+2.02.
# 판정선(|t|>2.5)에 못 미쳤지만 **3판 전부가 기준선 평균을 넘었고 산포는 절반**이다.
# t 가 모자란 건 자유도 부족 탓이 크다. 시드 3,4 를 추가해 5v5 짝지은 비교로 확정한다.
#
# 같은 날 synth 가 CV 에서 t=7.8 이었는데 e2e 에서 -0.0205 로 뒤집혔다 — 위양성이 없는
# 하니스는 e2e 를 예측 못 한다는 게 확인됐다. 그래서 geo 도 **e2e·시드 5판**으로만 판정한다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
GEOFEAT="$A/c34_feat_train_geoarc.json"
log(){ echo "[e9 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== geo 시드 3,4 추가 (0~2 는 E7 에서 완료) ==="
for SD in 3 4; do
  TAG="e7_geo_s${SD}"
  [ -f "$A/c5_eval_test_${TAG}.json" ] && { log "  $TAG 있음"; continue; }
  log "  test $TAG"
  CLF_SEED=$SD $PY -u c5_location_v2.py eval --train-feat "$GEOFEAT" --use-geo --split test \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
    --aneurysm-pred-dir "$P/aneu_test_probavgf" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$TAG" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
done

log "=== 5v5 짝지은 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
K=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def load(pat):
    out={}
    for f in glob.glob(os.path.join(A,pat)):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1))
        d=json.load(open(f)); out[sd]=(d["adjusted_div_present"],d["official_div52"])
    return out
g=load("c5_eval_test_e7_geo_s?.json"); b=load("c5_eval_test_e4_rf_s?.json")
sds=sorted(set(g)&set(b))
print(f"공통 시드 {sds}")
comp=lambda o:(o['PRECISION']+o['RECALL']+o['MCC']+o['DICE']+o['VOLSIM']+1-o['HD95'])/6
print(f"\n{'시드':>4}{'geo cov.MCC':>13}{'현행 cov.MCC':>14}{'차이':>10}")
for s in sds:
    print(f"{s:>4}{g[s][0]['MCC']:>13.4f}{b[s][0]['MCC']:>14.4f}{g[s][0]['MCC']-b[s][0]['MCC']:>+10.4f}")
print(f"\n{'지표':<10}{'geo':>18}{'현행':>18}{'짝지은 차이':>16}{'유리시드':>9}")
for k in K:
    ga=np.array([g[s][0][k] for s in sds]); ba=np.array([b[s][0][k] for s in sds])
    d=ga-ba; w=int((d>0).sum()) if k!='HD95' else int((d<0).sum())
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0
    print(f"{k:<10}{ga.mean():>10.4f}±{ga.std():<7.4f}{ba.mean():>10.4f}±{ba.std():<7.4f}"
          f"{d.mean():>+9.4f}(t{t:+.1f}){w:>6}/{len(sds)}")
ga=np.array([g[s][0]['MCC'] for s in sds]); ba=np.array([b[s][0]['MCC'] for s in sds])
oa=np.array([g[s][1]['MCC'] for s in sds]); ob=np.array([b[s][1]['MCC'] for s in sds])
d=ga-ba; t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
print(f"\n  cov.MCC  geo {ga.mean():.4f}±{ga.std():.4f}  vs  현행 {ba.mean():.4f}±{ba.std():.4f}")
print(f"  off.MCC  geo {oa.mean():.4f}±{oa.std():.4f}  vs  현행 {ob.mean():.4f}±{ob.std():.4f}")
print(f"  짝지은 차이 {d.mean():+.4f} ± {d.std(ddof=1):.4f}   t = {t:+.2f}   이긴 시드 {int((d>0).sum())}/{len(d)}")
print(f"  [판정] {'★ 채택 — 새 최고' if d.mean()>0 and t>2.5 else ('열세' if d.mean()<0 and t<-2.5 else '판정 불가 — 기준선 유지')}")
PYEOF

log "=== E1b 재개 ==="
setsid nohup env TOPANEU_ROOT="$R" bash "$S/chain_e1b.sh" >> "$E/e1b_chain.log" 2>&1 < /dev/null & disown
setsid nohup env TOPANEU_ROOT="$R" bash "$S/watchdog_e.sh" >> "$E/watchdog_e.log" 2>&1 < /dev/null & disown
log "=== 완료 ==="
