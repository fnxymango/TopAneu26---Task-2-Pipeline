#!/usr/bin/env bash
# W1c/W1d — Maha 혼합 후속 (2026-08-25).
#
# W1b 결과: w 0.15->0.60 이 **단조 증가**다 (Δ +0.0053 / +0.0060 / +0.0183 / +0.0250).
#   w=0.45 가 t=2.93, 5/5 로 판정선을 통과했지만 곡선이 안 꺾인 게 걸린다.
#   좌표 단독 분류기는 우리 RF 보다 약하므로(jslee 실측 top1 38.6% vs 우리 73%)
#   어딘가에서 반드시 꺾여야 한다. 안 꺾이면 내가 재는 게 Maha 이득이 아니라
#   **pos 블록 가중치 재조정**일 수 있다 — 그건 Q-CPU 에서 이미 졌던 축이라 모순이 된다.
#
#   W1c  w=0.75 / 0.90 (test)  -> 꺾이는 지점을 찾는다
#   W1d  w=0.45 / 0.60 (292 OOF) -> 진짜 판정. E10 통짜가 test t=4.68 -> 292 t=1.21 로
#        무너진 전례가 있어 이 관문을 반드시 거친다.
#
# 학습 2건이 GPU 를 쓰고 데이터로더가 CPU 를 먹는 중이라 병렬을 낮게 잡고 OMP 도 묶는다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/e11_feat_hyb_ov.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
STATUS="$E/w2_status.log"; CAP="${W2CAP:-3}"
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
st(){ echo "[STEP] $*" | tee -a "$STATUS"; }

run(){ # $1=split $2=tag $3=maha_w
  local sp=$1 tag=$2 w=$3 PIDS=() EXTRA="" VD VB AD
  if [ "$sp" = train ]; then VD=vespp_train; VB=vespp_train; AD="$P/aneu_train_ooff"
                             EXTRA="--cv-splits $SP720"
  else VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"; fi
  for SD in 0 1 2 3 4; do
    [ -f "$A/c5_eval_${sp}_${tag}_s${SD}.json" ] && continue
    TOPANEU_MAHA_W=$w CLF_SEED=$SD OMP_NUM_THREADS=2 nohup $PY -u c5_location_v2.py eval \
      --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 $EXTRA \
      --tag "${tag}_s${SD}" > "$E/w2_${tag}_s${SD}.log" 2>&1 &
    PIDS+=($!)
    [ "${#PIDS[@]}" -ge "$CAP" ] && { for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); }
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
}

st "W1c 시작 · test · Maha w=0.75 / 0.90 (곡선이 꺾이는지)"
run test w1_maha75 0.75
run test w1_maha90 0.90
st "W1c 완료"
$PY -u p_weak_report.py e11_hyb_ov w1_maha45 w1_maha60 w1_maha75 w1_maha90 2>&1 | tee -a "$STATUS"

st "W1d 시작 · 292 OOF 중재 · w=0.45 / 0.60 (케이스 3.5배, 오래 걸림)"
run train w1d_maha45 0.45
run train w1d_maha60 0.60
st "W1d 완료"

st "최종 판정"
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
def cmp(sp,base_tag,items,title):
    base=load(sp,base_tag)
    if not base: print(f"\n{title}: 기준 없음"); return
    a=np.array([base[s] for s in sorted(base)])
    print(f"\n{title}  기준 n={len(a)} {a.mean():.4f} ± {a.std():.4f}")
    print(f"  {'설정':<14}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ':>10}{'t':>7}{'승':>6}  판정")
    for tag,lab in items:
        g=load(sp,tag); sds=sorted(set(g)&set(base))
        if len(sds)<2: print(f"  {lab:<14} 미완 n={len(sds)}"); continue
        ga=np.array([g[s] for s in sds]); ba=np.array([base[s] for s in sds]); d=ga-ba
        t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
        nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
        print(f"  {lab:<14}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+10.4f}"
              f"{t:>+7.2f}{nw:>4}/{len(d)}  {'★통과' if ok else '기각'}")
cmp("test","e11_hyb_ov",[("w1_maha15","w=0.15"),("w1_maha30","w=0.30"),("w1_maha45","w=0.45"),
    ("w1_maha60","w=0.60"),("w1_maha75","w=0.75"),("w1_maha90","w=0.90")],"[test 83]")
cmp("train","e1b2_ov",[("w1d_maha45","w=0.45"),("w1d_maha60","w=0.60")],"[292 OOF]")
print("\n  채택 조건: test 통과 **그리고** 292 통과. 하나라도 미달이면 현행 유지.")
PYEOF
st "W1c/W1d 끝"
