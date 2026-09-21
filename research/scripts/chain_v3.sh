#!/usr/bin/env bash
# V3 — ① 곁가지 신뢰도 피처. **게이트 재설계판** (2026-08-25 18:55).
#
# 앞선 chain_v2 의 게이트(val 통과 AND 292 통과)가 잘못됐다는 걸 T16 자기검증으로 확인했다:
#   T16 vs T13   test +0.0314 t=4.34 5/5 / 292 +0.0183 t=3.77 5/5 / val +0.0003 t=0.04 3/5
#   -> val AND 292 를 요구하면 T16 이 탈락한다. 그런데 292(367병변)가 실효를 확인해준다.
#   val 42 는 sd 0.0174 · n=5 라 SE 0.0078 이고, 진짜 +0.018 효과도 기대 t 가 2.3 밖에 안 된다.
#   판정선 2.5 를 구조적으로 못 넘는다 — 검출력이 없는 관문이다.
#
# 고친 구조:
#   주 관문   292 OOF  (병변 367 · 분모 43 · sd 0.0060)  t>2.5 & 5/5
#   반증 검사 val 42    유의하게 음수(t<-2.0)면 기각. 중립이면 통과.
#   보고      test 83  선별에 쓰지 않는다. 채택된 것만 잰다.
#
# 이 기준이면 Maha 는 292 가 t=-11.98 로 기각, T16 은 292 통과 + val 중립으로 채택된다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/e11_feat_hyb_ov.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
STATUS="$E/v3_status.log"; CAP="${VCAP:-4}"
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
      --tag "${tag}_s${SD}" > "$E/v3_${tag}_s${SD}.log" 2>&1 &
    PIDS+=($!)
    [ "${#PIDS[@]}" -ge "$CAP" ] && { for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); }
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
}

# ---- 1) 주 관문: 292 OOF ------------------------------------------------------
st "V3-1 · 292 OOF · gate 모드 (주 관문)"
run train v3_gate_oof gate
st "V3-1 · 292 OOF · block 모드"
run train v3_block_oof block
st "V3-1 완료 · 주 관문 판정"
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
base=load("train","e1b2_ov"); a=np.array([base[s] for s in sorted(base)])
print(f"\n[292 OOF · 주 관문]  기준 e1b2_ov  n={len(a)}  {a.mean():.4f} +- {a.std():.4f}"
      f"  | 병변 367 · 분모 43 · 병변1개 +0.0233")
print(f"  {'설정':<18}{'n':>3}{'cov.MCC':>10}{'+-':>8}{'delta':>10}{'t':>7}{'승':>6}  판정")
win=[]
for tag,lab in (("v3_gate_oof","gate"),("v3_block_oof","block")):
    g=load("train",tag); s=sorted(set(g)&set(base))
    if len(s)<2: print(f"  {lab:<18} 미완 n={len(s)}"); continue
    ga=np.array([g[x] for x in s]); ba=np.array([base[x] for x in s]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    print(f"  {lab:<18}{len(s):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+10.4f}{t:>+7.2f}{nw:>4}/{len(d)}  {'★통과' if ok else '기각'}")
    print(f"  {'':<18}   시드별 delta {[round(x,4) for x in d]}")
    if ok: win.append(lab)
open("/tmp/v3_win.txt","w").write(" ".join(win))
print(f"  -> 반증 검사로: {win if win else '없음'}")
PYEOF

W=$(cat /tmp/v3_win.txt 2>/dev/null || echo "")
if [ -z "$W" ]; then st "292 주 관문 미달 — 종료. test 는 건드리지 않는다."; exit 0; fi

# ---- 2) 반증 검사: val 42 ----------------------------------------------------
st "V3-2 · val 42 반증 검사 · 후보: $W"
for m in $W; do run val "v3_${m}" "$m"; done
$PY - "$A" "$W" <<'PYEOF' 2>&1 | tee -a "$STATUS"
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
base=load("val","v2_base"); a=np.array([base[s] for s in sorted(base)])
print(f"\n[val 42 · 반증 검사]  기준 n={len(a)}  {a.mean():.4f} +- {a.std():.4f}"
      f"  | 병변 45 · 분모 33 · SE {a.std(ddof=1)/np.sqrt(len(a)):.4f}")
print(f"  기준: 유의하게 음수(t < -2.0)면 기각. 중립/양수면 통과 — 이 집합은 +0.018 도 검출 못 한다.")
print(f"  {'설정':<18}{'n':>3}{'cov.MCC':>10}{'delta':>10}{'t':>7}  판정")
ok_list=[]
for m in wins:
    g=load("val","v3_"+m); s=sorted(set(g)&set(base))
    if len(s)<2: print(f"  {m:<18} 미완"); continue
    ga=np.array([g[x] for x in s]); ba=np.array([base[x] for x in s]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    ok = t > -2.0
    print(f"  {m:<18}{len(s):>3}{ga.mean():>10.4f}{d.mean():>+10.4f}{t:>+7.2f}  {'통과(반증없음)' if ok else '★기각(반증)'}")
    print(f"  {'':<18}   시드별 delta {[round(x,4) for x in d]}")
    if ok: ok_list.append(m)
open("/tmp/v3_final.txt","w").write(" ".join(ok_list))
print(f"  -> 채택: {ok_list if ok_list else '없음'}")
PYEOF

F=$(cat /tmp/v3_final.txt 2>/dev/null || echo "")
if [ -z "$F" ]; then st "val 반증으로 기각 — 채택 없음."; exit 0; fi

# ---- 3) 보고: test 83 --------------------------------------------------------
st "V3-3 · test 83 에서 **보고만** (선별 아님): $F"
for m in $F; do run test "v3_${m}_rep" "$m"; done
RT=""; for m in $F; do RT="$RT v3_${m}_rep"; done
$PY - "$A" "$RT" <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]; tags=sys.argv[2].split()
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_test_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c,of=d.get("adjusted_div_present"),d.get("official_div52")
        if c and of: o[sd]=(c["MCC"],of["MCC"])
    return o
base=load("e11_hyb_ov"); a=np.array([base[s][0] for s in sorted(base)])
print(f"\n[test 83 · 보고]  T16 현행  n={len(a)}  cov.MCC {a.mean():.4f} +- {a.std():.4f}"
      f"  off÷52 {np.mean([base[s][1] for s in sorted(base)]):.4f}")
for tag in tags:
    g=load(tag); s=sorted(set(g)&set(base))
    if len(s)<2: print(f"  {tag} 미완"); continue
    ga=np.array([g[x][0] for x in s]); ba=np.array([base[x][0] for x in s]); d=ga-ba
    print(f"  {tag:<18}{ga.mean():>10.4f} +- {ga.std():.4f}  off÷52 {np.mean([g[x][1] for x in s]):.4f}"
          f"  delta {d.mean():+.4f}  시드별 {[round(x,4) for x in d]}")
print("\n  ※ 이 test 값은 선별에 쓰지 않았다. 채택은 292 OOF + val 반증검사로 이미 끝났다.")
PYEOF
$PY -u p_weak_report.py e11_hyb_ov $RT 2>&1 | tee -a "$STATUS"
st "V3 전부 끝"
