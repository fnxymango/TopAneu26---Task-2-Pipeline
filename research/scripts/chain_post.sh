#!/usr/bin/env bash
# POST — 병변기록이 나오면 자동으로 이어지는 후속 분석 (2026-08-25).
#   1) 오라클 분해   라벨교정 vs 환각제거 vs 미검출회수 — 어디에 얼마를 걸어야 하나
#   2) 모달리티 분해 CT vs MR — 저쪽 gold 감사에서 CT 0.62 / MR 0.80 격차 확인됨.
#                    우리는 e2e 를 모달리티로 쪼개본 적이 없다.
# 입력이 준비될 때까지 파일로만 기다린다(pgrep 안 씀).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
STATUS="$E/post_status.log"
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
st(){ echo "[STEP] $*" | tee -a "$STATUS"; }

st "POST 대기 · 병변기록 3시드 생성 기다리는 중"
for i in $(seq 1 240); do
  n=$(ls "$A"/p_lesion_e11_hyb_ov_s?.json 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && break
  sleep 60
done
n=$(ls "$A"/p_lesion_e11_hyb_ov_s?.json 2>/dev/null | wc -l)
[ "$n" -lt 1 ] && { echo "[FAIL] 병변기록 없음" | tee -a "$STATUS"; exit 1; }
st "POST 시작 · 병변기록 ${n}시드 확보"

st "오라클 분해 + 모달리티 분해"
$PY - "$A" "$n" <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json,glob,os,sys,collections
import numpy as np
A=sys.argv[1]
def macro_mcc(recs,present,n_by_case):
    by=collections.defaultdict(collections.Counter)
    for r in recs:
        if r["truth"]: by[r["case"]]["g_"+r["truth"]]+=1
        if r["pred"]:  by[r["case"]]["p_"+r["pred"]]+=1
    tot={c:[0,0,0,0] for c in present}
    for case,cnt in by.items():
        na=n_by_case.get(case,0)
        for c in present:
            g=cnt.get("g_"+c,0); p=cnt.get("p_"+c,0)
            tp=min(p,g); fp=max(0,p-g); fn=max(0,g-p); tn=na-(tp+fn)
            t=tot[c]; t[0]+=tp; t[1]+=fp; t[2]+=fn; t[3]+=tn
    vals=[]
    for c in present:
        tp,fp,fn,tn=tot[c]
        num=tp*tn-fn*fp
        den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        vals.append(num/den if den>0 else 0.0)
    return float(np.mean(vals))

def scen(recs):
    fix=[dict(r) for r in recs]
    for r in fix:
        if r["truth"] and r["pred"]: r["pred"]=r["truth"]
    noh=[r for r in recs if r["truth"] or not r["pred"]]
    both=[r for r in fix if r["truth"] or not r["pred"]]
    det=[dict(r) for r in recs]
    for r in det:
        if r["truth"] and not r["pred"]: r["pred"]=r["truth"]   # 미검출까지 회수
    return {"라벨 전부 교정":fix,"환각 전부 제거":noh,"라벨+환각":both,"미검출까지 회수":det}

files=sorted(glob.glob(os.path.join(A,"p_lesion_e11_hyb_ov_s?.json")))
agg=collections.defaultdict(list); dec=collections.Counter(); mod=collections.defaultdict(lambda:[0,0,0,0])
for f in files:
    recs=json.load(open(f))
    present=sorted({r["truth"] for r in recs if r["truth"]})
    nbc=collections.Counter()
    for r in recs:
        nbc[r["case"]]+= 1 if r["truth"] else 0
    base=macro_mcc(recs,present,nbc); agg["현행 (재구성)"].append(base)
    for lab,rs in scen(recs).items(): agg[lab].append(macro_mcc(rs,present,nbc))
    for r in recs:
        k=("맞힘" if r["pred"]==r["truth"] else "라벨틀림") if (r["truth"] and r["pred"]) \
          else ("미검출" if r["truth"] else "환각")
        dec[k]+=1
        m="CT" if "_ct_" in r["case"] else "MR"
        i={"맞힘":0,"라벨틀림":1,"환각":2,"미검출":3}[k]; mod[m][i]+=1
S=len(files)
print(f"\n[오라클 분해] test 83 · 시드 {S}판 평균 · 분모 {len(present)}클래스")
print(f"  병변 분해(시드평균)  맞힘 {dec['맞힘']/S:.1f} · 라벨틀림 {dec['라벨틀림']/S:.1f} "
      f"· 환각 {dec['환각']/S:.1f} · 미검출 {dec['미검출']/S:.1f}")
b=np.mean(agg["현행 (재구성)"])
print(f"\n  {'시나리오':<22}{'cov.MCC':>10}{'Δ':>10}{'병변당':>10}")
print(f"  {'현행 (재구성)':<22}{b:>10.4f}")
for lab in ("라벨 전부 교정","환각 전부 제거","라벨+환각","미검출까지 회수"):
    v=np.mean(agg[lab]); n={"라벨 전부 교정":dec['라벨틀림'],"환각 전부 제거":dec['환각'],
        "라벨+환각":dec['라벨틀림']+dec['환각'],"미검출까지 회수":dec['미검출']}[lab]/S
    print(f"  {lab:<22}{v:>10.4f}{v-b:>+10.4f}{(v-b)/max(n,1):>+10.4f}")
print(f"\n[모달리티 분해] 시드평균")
print(f"  {'':<6}{'맞힘':>8}{'라벨틀림':>10}{'환각':>8}{'미검출':>8}{'GT':>6}{'라벨정확도':>11}")
for m in ("CT","MR"):
    a=[x/S for x in mod[m]]; gt=a[0]+a[1]+a[3]; det=a[0]+a[1]
    print(f"  {m:<6}{a[0]:>8.1f}{a[1]:>10.1f}{a[2]:>8.1f}{a[3]:>8.1f}{gt:>6.0f}"
          f"{(a[0]/det if det else 0):>11.3f}")
PYEOF
st "POST 끝"
