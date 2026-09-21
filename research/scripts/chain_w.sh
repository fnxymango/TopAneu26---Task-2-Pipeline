#!/usr/bin/env bash
# W — 약한 클래스 직격 큐 (2026-08-25).
#
# E16: 약한 클래스 22개가 test GT 병변 40/86(47%)을 차지한다. 손실은 검출 10 / 분류 12.
# 지배적 오류는 junction <-> trunk (혼동 상위 10쌍 중 6쌍). 약한 클래스의 12/22 가 ICA 계열.
#
# 집계 cov.MCC 는 시드 산포 ±0.02 라 개입 효과가 묻힌다. 그래서 **약한 클래스 TP** 를
# 함께 본다(p_weak_report.py). 채택 판정은 종전대로 집계 t>2.5 & 5/5 를 쓰되,
# 약한 클래스 TP 가 안 늘면 "왜 올랐는지 모르는 이득"으로 보고 기록에 남긴다.
#
# W1  좌표 Mahalanobis 혼합 (jslee §2.6b)  — CPU. ICA 직렬분절은 위치가 곧 판별자다.
# W2  junction/trunk 전용 이진 판별기      — CPU. E16 이 지목한 지배적 오류에 직격.
# (W3 이질적 확률앙상블 · W4 작은병변 역가중은 GPU 라 pjh 학습 뒤에 별도 체인)
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/e11_feat_hyb_ov.json"     # T16 기준
STATUS="$E/w_status.log"
CAP="${WCAP:-6}"
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
st(){ echo "[STEP] $*" | tee -a "$STATUS"; }
er(){ echo "[FAIL] $*" | tee -a "$STATUS"; exit 1; }

ev(){ # $1=tag $2=추가환경(공백구분) — test 5시드
  local tag=$1 envs=${2:-}
  local PIDS=()
  for SD in 0 1 2 3 4; do
    [ -f "$A/c5_eval_test_${tag}_s${SD}.json" ] && continue
    # shellcheck disable=SC2086
    env $envs CLF_SEED=$SD nohup $PY -u c5_location_v2.py eval \
      --train-feat "$FEAT" --split test \
      --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
      --aneurysm-pred-dir "$P/aneu_test_probavgf" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "${tag}_s${SD}" > "$E/w_${tag}_s${SD}.log" 2>&1 &
    PIDS+=($!)
    [ "${#PIDS[@]}" -ge "$CAP" ] && { for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); }
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
}

st "W1 시작 · 좌표 Mahalanobis 혼합 (w=0.15 / 0.30, 각 5시드)"
ev w1_maha15 "TOPANEU_MAHA_W=0.15"
ev w1_maha30 "TOPANEU_MAHA_W=0.30"
st "W1 완료"

st "W1 판정"
$PY - "$A" <<'PYEOF' | tee -a "$STATUS"
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_test_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c,ofc=d.get("adjusted_div_present"),d.get("official_div52")
        if c and ofc: o[sd]=(c["MCC"],ofc["MCC"])
    return o
base=load("e11_hyb_ov")
a=np.array([base[s][0] for s in sorted(base)])
print(f"\n[test 83] 기준 T16  n={len(a)}  cov.MCC {a.mean():.4f} ± {a.std():.4f}")
print(f"  {'설정':<16}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ':>10}{'t':>7}{'승':>6}  판정")
for tag,lab in (("w1_maha15","Maha w=0.15"),("w1_maha30","Maha w=0.30")):
    g=load(tag); sds=sorted(set(g)&set(base))
    if len(sds)<2: print(f"  {lab:<16} 데이터 부족 n={len(sds)}"); continue
    ga=np.array([g[s][0] for s in sds]); ba=np.array([base[s][0] for s in sds]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    print(f"  {lab:<16}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+10.4f}"
          f"{t:>+7.2f}{nw:>4}/5  {'★통과' if ok else '기각'}")
    print(f"  {'':<16}   시드별 Δ {[round(x,4) for x in d]}")
PYEOF
$PY -u p_weak_report.py e11_hyb_ov w1_maha15 w1_maha30 2>&1 | tee -a "$STATUS"
st "W1b 시작 · Maha w 스윕 확장 (0.45 / 0.60) — w=0.30 이 4/5 승이라 최적점 위치를 본다"
ev w1_maha45 "TOPANEU_MAHA_W=0.45"
ev w1_maha60 "TOPANEU_MAHA_W=0.60"
st "W1b 완료"

st "W2 시작 · junction/trunk 전용 이진 판별기 (좌우 통합, 7쌍)"
ev w2_pair "TOPANEU_PAIR=1"
st "W2b 시작 · W2 + Maha w=0.30 동시 적용"
ev w2_pair_maha "TOPANEU_PAIR=1 TOPANEU_MAHA_W=0.30"
st "W2 완료"

st "최종 판정"
$PY - "$A" <<'PYEOF2' | tee -a "$STATUS"
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_test_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c,ofc=d.get("adjusted_div_present"),d.get("official_div52")
        if c and ofc: o[sd]=(c["MCC"],ofc["MCC"])
    return o
base=load("e11_hyb_ov")
a=np.array([base[s][0] for s in sorted(base)])
print(f"\n[test 83] 기준 T16  n={len(a)}  cov.MCC {a.mean():.4f} +- {a.std():.4f}")
print(f"  {'설정':<22}{'n':>3}{'cov.MCC':>10}{'+-':>8}{'off':>8}{'delta':>9}{'t':>7}{'승':>6}  판정")
rows=[]
for tag,lab in (("w1_maha15","Maha w=0.15"),("w1_maha30","Maha w=0.30"),
                ("w1_maha45","Maha w=0.45"),("w1_maha60","Maha w=0.60"),
                ("w2_pair","쌍판별기"),("w2_pair_maha","쌍판별기+Maha0.3")):
    g=load(tag); sds=sorted(set(g)&set(base))
    if len(sds)<2: print(f"  {lab:<22} 데이터 부족 n={len(sds)}"); continue
    ga=np.array([g[s][0] for s in sds]); ba=np.array([base[s][0] for s in sds]); d=ga-ba
    oa=np.array([g[s][1] for s in sds])
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    rows.append((d.mean(),lab,len(sds),ga.mean(),ga.std(),oa.mean(),t,nw,ok,d))
for m,lab,n,gm,gs,om,t,nw,ok,d in sorted(rows,key=lambda r:-r[0]):
    print(f"  {lab:<22}{n:>3}{gm:>10.4f}{gs:>8.4f}{om:>8.4f}{m:>+9.4f}{t:>+7.2f}{nw:>4}/5  {'★통과' if ok else '기각'}")
    print(f"  {'':<22}   시드별 {[round(x,4) for x in d]}")
PYEOF2
$PY -u p_weak_report.py e11_hyb_ov w1_maha30 w1_maha45 w1_maha60 w2_pair w2_pair_maha 2>&1 | tee -a "$STATUS"
st "W 큐 전부 끝"
