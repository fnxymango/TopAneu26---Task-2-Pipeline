#!/usr/bin/env bash
# V — ① 곁가지 존재 신뢰도 피처 (2026-08-25).
#
# 진단 근거 셋이 한 점을 가리킨다:
#   (a) 라벨오류 17.7개 중 58.5% 가 "같은 혈관 인접분절 혼동", 최다 오답이 전부 -> Pcom-junction
#   (b) 예측 혈관에서 Pcom/AChA 만 GT 대비 1.31~1.46배 과대 등장 (다른 혈관은 1.0~1.1)
#       Pcom·AChA 는 해부학적으로 결손이 흔하다. 없는 걸 그리면 거리/중첩이 거짓말을 한다.
#   (c) top-k 곡선: 정답이 2등 안에 88.4%, 52등 밖 0.0%.
#       -> 피처에 정보가 없는 게 아니라 1·2등 판단만 틀린다. 재순위 여지 상한 +0.1839.
#
# W2(쌍 판별기)가 -0.0234 로 진 이유도 이걸로 설명된다: 같은 거짓 피처로 다시 판별하려 했다.
# 여기서는 판별기를 더 얹는 대신 **입력의 거짓말 자체를 누른다**.
#
#   gate   거리·중첩 피처에 conf 를 곱한다 (106차원 유지)
#   block  conf 36차원을 뒤에 붙인다 (142차원, RF 가 쓸지 스스로 정한다)
#
# 회귀검사 완료: VESCONF=off 에서 6지표 전부 오차 0 (seed0 MCC 0.385957).
# 채택 조건은 오늘 Maha 가 무너진 그대로 적용한다 — test 5/5 & t>2.5 **그리고** 292 OOF 통과.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/e11_feat_hyb_ov.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
STATUS="$E/v_status.log"; CAP="${VCAP:-3}"
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
st(){ echo "[STEP] $(TZ=Asia/Seoul date +%H:%M) $*" | tee -a "$STATUS"; }

run(){ # $1=split $2=tag $3=vesconf모드
  local sp=$1 tag=$2 mode=$3 PIDS=() EXTRA="" VD AD
  if [ "$sp" = train ]; then VD=vespp_train; AD="$P/aneu_train_ooff"; EXTRA="--cv-splits $SP720"
  else VD=vespp_test; AD="$P/aneu_test_probavgf"; fi
  for SD in 0 1 2 3 4; do
    [ -f "$A/c5_eval_${sp}_${tag}_s${SD}.json" ] && continue
    TOPANEU_VESCONF=$mode CLF_SEED=$SD OMP_NUM_THREADS=2 nohup $PY -u c5_location_v2.py eval \
      --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VD" --aneurysm-pred-dir "$AD" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 $EXTRA \
      --tag "${tag}_s${SD}" > "$E/v_${tag}_s${SD}.log" 2>&1 &
    PIDS+=($!)
    [ "${#PIDS[@]}" -ge "$CAP" ] && { for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); }
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
}

st "V 대기 · vesconf 표 생성 완료를 기다린다"
while [ ! -f "$A/vesconf_test.json" ] || [ ! -f "$A/vesconf_train.json" ]; do sleep 60; done
$PY -c "
import json
for s in ('train','test'):
    d=json.load(open('$A/vesconf_%s.json'%s)); print(f'  vesconf_{s}  {len(d)}케이스')" | tee -a "$STATUS"

st "V1 시작 · gate 모드 (거리·중첩에 conf 곱하기) · test 5시드"
run test v_gate gate
st "V1 완료"
st "V2 시작 · block 모드 (conf 36차원 추가) · test 5시드"
run test v_block block
st "V2 완료"

st "V 중간 판정 (test)"
$PY - "$A" <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def load(sp,tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_{sp}_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c=d.get("adjusted_div_present")
        if c: o[sd]=c["MCC"]
    return o
base=load("test","e11_hyb_ov"); a=np.array([base[s] for s in sorted(base)])
print(f"\n[test 83] 기준 T16  n={len(a)}  cov.MCC {a.mean():.4f} +- {a.std():.4f}")
print(f"  {'설정':<20}{'n':>3}{'cov.MCC':>10}{'+-':>8}{'delta':>10}{'t':>7}{'승':>6}  판정")
win=[]
for tag,lab in (("v_gate","gate 곱하기"),("v_block","block 추가")):
    g=load("test",tag); sds=sorted(set(g)&set(base))
    if len(sds)<2: print(f"  {lab:<20} 미완 n={len(sds)}"); continue
    ga=np.array([g[s] for s in sds]); ba=np.array([base[s] for s in sds]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    print(f"  {lab:<20}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+10.4f}{t:>+7.2f}{nw:>4}/{len(d)}  {'★통과' if ok else '기각'}")
    print(f"  {'':<20}   시드별 delta {[round(x,4) for x in d]}")
    if ok: win.append(tag)
open("/tmp/v_win.txt","w").write(" ".join(win))
print(f"\n  292 로 넘길 후보: {win if win else '없음'}")
PYEOF
$PY -u p_weak_report.py e11_hyb_ov v_gate v_block 2>&1 | tee -a "$STATUS"

WIN=$(cat /tmp/v_win.txt 2>/dev/null || echo "")
if [ -z "$WIN" ]; then
  st "test 관문 미달 — 292 안 돌린다. V 종료."
  exit 0
fi
st "V3 시작 · 292 OOF 중재 · 후보: $WIN"
for tag in $WIN; do
  mode=$([ "$tag" = v_gate ] && echo gate || echo block)
  run train "${tag}_oof" "$mode"
done
st "V3 완료 · 최종 판정"
$PY - "$A" "$WIN" <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]; wins=sys.argv[2].split()
def load(sp,tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_{sp}_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c=d.get("adjusted_div_present")
        if c: o[sd]=c["MCC"]
    return o
base=load("train","e1b2_ov"); a=np.array([base[s] for s in sorted(base)])
print(f"\n[292 OOF] 기준 e1b2_ov  n={len(a)}  cov.MCC {a.mean():.4f} +- {a.std():.4f}")
print(f"  {'설정':<20}{'n':>3}{'cov.MCC':>10}{'+-':>8}{'delta':>10}{'t':>7}{'승':>6}  판정")
for tag in wins:
    g=load("train",tag+"_oof"); sds=sorted(set(g)&set(base))
    if len(sds)<2: print(f"  {tag:<20} 미완 n={len(sds)}"); continue
    ga=np.array([g[s] for s in sds]); ba=np.array([base[s] for s in sds]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    print(f"  {tag:<20}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+10.4f}{t:>+7.2f}{nw:>4}/{len(d)}  {'★통과' if ok else '기각'}")
    print(f"  {'':<20}   시드별 delta {[round(x,4) for x in d]}")
print("\n  채택 조건: test 통과 **그리고** 292 통과. 오늘 Maha 가 test +0.0183 -> 292 -0.0132 로 무너졌다.")
PYEOF
st "V 전부 끝"
