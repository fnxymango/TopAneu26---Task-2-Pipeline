#!/usr/bin/env bash
# V2 — ① 곁가지 신뢰도 피처 · **test 를 선별에 쓰지 않는 첫 실험** (2026-08-25).
#
# 왜 바꾸나: 지금까지 test 83 이 선별 관문이면서 동시에 보고 점수였다. Q-CPU 18개,
#   E계열, W계열까지 40개 넘는 설정을 test 에 대고 걸러왔으니 0.3790 은 그만큼 부풀어 있다.
#   분할을 확인해보니 val 42 는 검출기 D722 folds 0~4 도, 혈관 V4-2 fold0 도 학습에 안 썼다.
#   그래서 선별을 val 42 + 292 OOF 로 옮기고 **test 는 채택된 것 하나만 보고용으로** 잰다.
#
#   V0  val 42 기준선 (VESCONF=off) — 정확히 짝이 맞는 기준선을 직접 만든다
#   V1  val 42 · gate  (거리·중첩에 conf 곱하기, 106차원)
#   V2  val 42 · block (conf 36차원 추가, 142차원)
#   V3  val 통과분만 292 OOF
#   V4  두 관문 다 통과한 것만 test 83 에서 **보고**  (선별 아님)
#
# 근거 셋: 라벨오류 58.5% 가 같은혈관 인접분절 혼동, 최다 오답 전부 -> Pcom-junction /
#   예측혈관에서 Pcom·AChA 만 GT 대비 1.31~1.46배 과대등장 / top-k 곡선 2등 안 88.4%.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/e11_feat_hyb_ov.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
STATUS="$E/v2_status.log"; CAP="${VCAP:-3}"
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
st(){ echo "[STEP] $(TZ=Asia/Seoul date +%H:%M) $*" | tee -a "$STATUS"; }

run(){ # $1=split $2=tag $3=vesconf모드
  local sp=$1 tag=$2 mode=$3 PIDS=() EXTRA="" VD BD AD
  case "$sp" in
    train) VD=vespp_train; BD=vespp_train; AD="$P/aneu_train_ooff"; EXTRA="--cv-splits $SP720";;
    val)   VD=vespp_val;   BD=val_pred;    AD="$P/aneu_val_probavgf";;
    test)  VD=vespp_test;  BD=vespp_test;  AD="$P/aneu_test_probavgf";;
  esac
  for SD in 0 1 2 3 4; do
    [ -f "$A/c5_eval_${sp}_${tag}_s${SD}.json" ] && continue
    TOPANEU_VESCONF=$mode CLF_SEED=$SD OMP_NUM_THREADS=2 nohup $PY -u c5_location_v2.py eval \
      --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$BD" --aneurysm-pred-dir "$AD" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 $EXTRA \
      --tag "${tag}_s${SD}" > "$E/v2_${tag}_s${SD}.log" 2>&1 &
    PIDS+=($!)
    [ "${#PIDS[@]}" -ge "$CAP" ] && { for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); }
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
}

judge(){ # $1=split $2=기준tag $3=제목 $4..=후보tag  -> 통과한 tag 를 /tmp/v2_win_$1.txt 에
  local sp=$1 base=$2 title=$3; shift 3
  $PY - "$A" "$sp" "$base" "$title" "$@" <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json,glob,os,sys,re
import numpy as np
A,sp,base_tag,title=sys.argv[1:5]; cands=sys.argv[5:]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_{sp}_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c=d.get("adjusted_div_present")
        if c: o[sd]=(c["MCC"],d.get("n_present_classes_in_split"),d.get("n_lesions_predicted"))
    return o
base=load(base_tag)
if not base:
    print(f"\n{title}: 기준({base_tag}) 없음"); open(f"/tmp/v2_win_{sp}.txt","w").write(""); raise SystemExit
a=np.array([base[s][0] for s in sorted(base)])
ncl=base[sorted(base)[0]][1]; nles=base[sorted(base)[0]][2]
print(f"\n{title}  기준 n={len(a)} {a.mean():.4f} +- {a.std():.4f}"
      f"  | 분모 {ncl}클래스 · 예측병변 {nles} · 병변 1개 = {1.0/ncl:+.4f}")
print(f"  {'설정':<18}{'n':>3}{'cov.MCC':>10}{'+-':>8}{'delta':>10}{'t':>7}{'승':>6}  판정")
win=[]
for tag in cands:
    g=load(tag); sds=sorted(set(g)&set(base))
    if len(sds)<2: print(f"  {tag:<18} 미완 n={len(sds)}"); continue
    ga=np.array([g[s][0] for s in sds]); ba=np.array([base[s][0] for s in sds]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    print(f"  {tag:<18}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+10.4f}"
          f"{t:>+7.2f}{nw:>4}/{len(d)}  {'★통과' if ok else '기각'}")
    print(f"  {'':<18}   시드별 delta {[round(x,4) for x in d]}")
    if ok: win.append(tag)
open(f"/tmp/v2_win_{sp}.txt","w").write(" ".join(win))
print(f"  -> 다음 관문으로: {win if win else '없음'}")
PYEOF
}

st "V2 대기 · vesconf 표 생성 완료를 기다린다"
while [ ! -f "$A/vesconf_test.json" ] || [ ! -f "$A/vesconf_train.json" ]; do sleep 45; done
$PY -c "
import json
for s in ('train','test'):
    d=json.load(open('$A/vesconf_%s.json'%s)); print(f'  vesconf_{s}  {len(d)}케이스')" | tee -a "$STATUS"

# val 42 케이스의 conf 는 test 표에 없다 — vespp_val 로 따로 만든다
if [ ! -f "$A/vesconf_val.json" ]; then
  st "val 42 conf 표 생성"
  $PY - <<'PYEOF' 2>&1 | tee -a "$STATUS"
import os,sys,json
sys.path.insert(0,os.getcwd())
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L
R=L.TOPANEU_ROOT; P=R/"experiments/_c1_realpred"; A=R/"code/sblee/nnunet/analysis"
NAMES={v:k for k,v in json.load(open(R/"dataset/TopAneu/vessel_mapping.json"))["labels"].items()}
ST=np.ones((3,3,3),bool); ref=json.load(open(A/"vesconf_ref.json"))
_,val_ids,_=L.case_ids_by_split(); out={}
for cid in val_ids:
    f=P/"vespp_val"/f"{cid}.nii.gz"
    if not f.exists(): continue
    a=np.asanyarray(nib.load(str(f)).dataobj).astype(np.int32,copy=False)
    boxes=ndi.find_objects(a,max_label=36); d={}
    for c in range(1,37):
        sl=boxes[c-1]
        if sl is None: continue
        lab,k=ndi.label(a[sl]==c,structure=ST)
        if not k: continue
        v=int(np.bincount(lab.ravel())[1:].max())
        nm=NAMES.get(c,str(c))
        d[nm]=round(min(2.0, v/max(ref.get(nm,1.0),1.0)),4)
    out[cid]=d
json.dump(out,open(A/"vesconf_val.json","w"))
print(f"  vesconf_val {len(out)}케이스")
PYEOF
fi

st "V0 · val 42 기준선 (VESCONF=off, 5시드)"
run val v2_base off
st "V1 · val 42 gate 모드"
run val v2_gate gate
st "V2 · val 42 block 모드"
run val v2_block block
judge val v2_base "[val 42 · 선별 1관문]" v2_gate v2_block

W=$(cat /tmp/v2_win_val.txt 2>/dev/null || echo "")
if [ -z "$W" ]; then st "val 관문 미달 — 292 안 돌린다. V2 종료."; exit 0; fi

st "V3 · 292 OOF 중재 · 후보: $W"
for tag in $W; do
  mode=$([ "$tag" = v2_gate ] && echo gate || echo block)
  run train "${tag}_oof" "$mode"
done
OOFT=""; for tag in $W; do OOFT="$OOFT ${tag}_oof"; done
judge train e1b2_ov "[292 OOF · 선별 2관문]" $OOFT

W2=$(cat /tmp/v2_win_train.txt 2>/dev/null || echo "")
if [ -z "$W2" ]; then st "292 관문 미달 — 채택 없음. test 는 건드리지 않는다."; exit 0; fi

st "V4 · 두 관문 통과 → test 83 에서 **보고만** (선별 아님): $W2"
for tag in $W2; do
  bt=${tag%_oof}
  mode=$([ "$bt" = v2_gate ] && echo gate || echo block)
  run test "${bt}_rep" "$mode"
done
REPT=""; for tag in $W2; do bt=${tag%_oof}; REPT="$REPT ${bt}_rep"; done
judge test e11_hyb_ov "[test 83 · 보고용 (선별에 안 씀)]" $REPT
$PY -u p_weak_report.py e11_hyb_ov $REPT 2>&1 | tee -a "$STATUS"
st "V2 전부 끝"
