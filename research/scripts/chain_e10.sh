#!/usr/bin/env bash
# E10 — 학습 피처의 **혈관을 참조 → 예측**으로 바꾼다 (2026-08-19).
#
# 문제(2026-08-19 사용자 지적): 지금 분류기는
#     학습  GT 병변 + **참조** 혈관(labelsTr)
#     추론  예측 병변 + **예측** 혈관(V4-2 출력)
# 로 혈관 쪽 분포가 어긋나 있다. 병변 쪽은 학습에 GT 를 쓰는 게 맞다(라벨이 거기서 나온다).
# 어긋난 건 혈관뿐이고, 피처 112차원이 전부 혈관 기준이라 영향이 크다.
#     참조 혈관: Pcom·AChA 항상 있음 · 조각 없음
#     예측 혈관: Pcom 53% · AChA 61% · 끊김/잡가지 있음
# 학습 때 "Pcom 까지 4.2mm" 를 보고 배운 분류기가 추론에서는 절반의 경우 그 칸이 결측이다.
# 정답 유출이 아니라 분포 불일치(covariate shift)이고, 손해다.
#
# 측지거리(geo)가 특히 취약하다 — 혈관을 따라가는 거리라 한 군데만 끊겨도 값이 무너진다.
# 유클리드 거리는 완만히 나빠지지만 측지는 절벽이다.
#
# 재료는 E1 이 방금 만들어놨다: vespp_train 292/292 + 분기점 그래프 292/292.
# 한계 — vespp_train 은 V4-2 fold0 이 만든 것이고 그 폴드가 292 를 학습했다. 그래서
# 실제 test 예측보다 깨끗하다(참조와 test예측의 중간쯤). 불일치를 절반쯤만 줄인다.
#
# 판정: 현행 기준선 rf β0.5 τ0.5 = 0.3477 ± 0.0199 (5판) 과 **시드 짝지어** 비교.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
PFEAT="$A/c10_feat_train_predves.json"          # 예측혈관 기반 학습 피처
PGEO="$A/e10_feat_train_predves_geo.json"       # + 측지/호위치 블록
log(){ echo "[e10 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== E9(geo 확정) 완료 대기 ==="
for i in $(seq 1 40); do
  [ "$(ls "$A"/c5_eval_test_e7_geo_s?.json 2>/dev/null | wc -l)" -ge 5 ] && { log "  E9 완료"; break; }
  log "  대기 $((i*3))분"; sleep 180
done

# ── STEP1: 예측혈관으로 학습 피처 재추출 (병변은 GT 그대로) ───────────
log "=== STEP1: 학습 피처 재추출 (혈관=vespp_train) ==="
if [ ! -f "$PFEAT" ]; then
  $PY -u c5_location_v2.py build --split train \
    --vessel-dir "$P/vespp_train" --bp-dir "$BP/vespp_train" --out "$PFEAT" 2>&1 | tail -3
fi
[ -f "$PFEAT" ] || { log "피처 생성 실패 — 중단"; exit 1; }

# C10 랜드마크 좌표(pos) 블록 채우기 — 기존 학습 피처와 같은 구성으로 맞춘다
if ! $PY -c "import json,sys; d=json.load(open('$PFEAT')); sys.exit(0 if d and 'pos' in d[0] else 1)" 2>/dev/null; then
  log "  pos 블록 추가"
  $PY -u c10_landmark_coords.py --feat "$PFEAT" --bp-dir "$BP/vespp_train" \
      --split train --out "$PFEAT.tmp" 2>&1 | tail -2 && mv "$PFEAT.tmp" "$PFEAT"
fi
$PY - "$PFEAT" "$A/c10_feat_train.json" <<'PYEOF'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
print(f"[e10] 예측혈관 학습피처 {len(a)}병변 (참조혈관판 {len(b)}병변) · 키 {sorted(a[0].keys())}")
PYEOF

# ── STEP2: 측지/호위치 블록도 예측혈관에서 채운다 ─────────────────────
log "=== STEP2: geo/arc 블록 채우기 (예측혈관) ==="
if [ ! -f "$PGEO" ]; then
  $PY - "$PFEAT" "$P/vespp_train" "$BP/vespp_train" "$PGEO" <<'PYEOF' 2>&1 | tail -4
import json,sys,collections
import numpy as np, nibabel as nib
sys.path.insert(0,".")
import d9xx_lib as L, c5_location_v2 as C5, c34_arc_position as C34
rows=json.load(open(sys.argv[1])); vdir,bdir,out=sys.argv[2],sys.argv[3],sys.argv[4]
ves_names=L.vessel_dense_names()
by=collections.defaultdict(list)
for r in rows: by[r["case"]].append(r)
ok=0
for i,(cid,rs) in enumerate(by.items(),1):
    try:
        vi=nib.load(f"{vdir}/{cid}.nii.gz"); ves=np.asanyarray(vi.dataobj)
        sp=np.array(vi.header.get_zooms()[:3],dtype=float)
        nodes=C5.load_bp(bdir,cid)
        C34.fill(rs,ves,sp,nodes,want_geo=True,want_arc=True); ok+=1
    except Exception as e:
        print(f"  {cid} 실패 {e}")
    if i%50==0: print(f"  {i}/{len(by)}",flush=True)
json.dump(rows,open(out,"w"),ensure_ascii=False)
n=sum(1 for r in rows if r.get("geo"))
print(f"[e10] geo 채워진 병변 {n}/{len(rows)} · 케이스 {ok}/{len(by)} -> {out}")
PYEOF
fi

# ── STEP3: test 평가 (시드 5판) ───────────────────────────────────────
ev(){ # $1=tag $2=feat $3=extra
  local tag=$1 feat=$2; shift 2
  for SD in 0 1 2 3 4; do
    local T="${tag}_s${SD}"
    [ -f "$A/c5_eval_test_${T}.json" ] && continue
    log "  test $T"
    CLF_SEED=$SD $PY -u c5_location_v2.py eval --train-feat "$feat" --split test \
      --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
      --aneurysm-pred-dir "$P/aneu_test_probavgf" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "$T" "$@" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
  done
}
log "=== STEP3: test 시드 5판 ==="
ev e10_predves "$PFEAT"
[ -f "$PGEO" ] && ev e10_predves_geo "$PGEO" --use-geo

log "=== STEP4: 짝지은 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re,collections
import numpy as np
A=sys.argv[1]; K=['PRECISION','RECALL','MCC','DICE','VOLSIM','HD95']
def load(pat):
    o={}
    for f in glob.glob(os.path.join(A,pat)):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        o[sd]=(d["adjusted_div_present"],d["official_div52"])
    return o
base=load("c5_eval_test_e4_rf_s?.json")
cands=[("e10_predves","예측혈관 학습"),("e10_predves_geo","예측혈관+geo"),
       ("e7_geo","참조혈관+geo")]
print(f"\n{'설정':<18}{'n':>3}{'cov.MCC':>11}{'±':>8}{'off.MCC':>10}{'짝지은Δ':>10}{'t':>7}{'승':>6}  판정")
b=np.array([base[s][0]['MCC'] for s in sorted(base)])
ob=np.array([base[s][1]['MCC'] for s in sorted(base)])
print(f"{'현행(기준선)':<18}{len(b):>3}{b.mean():>11.4f}{b.std():>8.4f}{ob.mean():>10.4f}"
      f"{'—':>10}{'—':>7}{'—':>6}")
for tag,lab in cands:
    g=load(f"c5_eval_test_{tag}_s?.json")
    sds=sorted(set(g)&set(base))
    if not sds: continue
    ga=np.array([g[s][0]['MCC'] for s in sds]); ba=np.array([base[s][0]['MCC'] for s in sds])
    oa=np.array([g[s][1]['MCC'] for s in sds])
    d=ga-ba; t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 and d.std(ddof=1)>0 else 0
    v="★ 채택" if d.mean()>0 and t>2.5 else ("열세" if d.mean()<0 and t<-2.5 else "판정 불가")
    print(f"{lab:<18}{len(sds):>3}{ga.mean():>11.4f}{ga.std():>8.4f}{oa.mean():>10.4f}"
          f"{d.mean():>+10.4f}{t:>+7.2f}{int((d>0).sum()):>4}/{len(d)}  {v}")
print("\n  Δ·t·승 은 **같은 시드끼리 짝지은** 비교다. 표집 노이즈가 상쇄되어 절대 산포보다 정밀하다.")
PYEOF
log "=== 완료 ==="
