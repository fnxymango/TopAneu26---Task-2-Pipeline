#!/usr/bin/env python3
"""회수 가능한 몫이 어디에 있나 — test GT 병변별로 (학습표본수 × 입력충실도) 격자에 정확도를 깐다."""
import json, os, collections, numpy as np, nibabel as nib
from scipy import ndimage
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; A=f"{R}/code/sblee/nnunet/analysis"
E=f"{R}/experiments"; PRED=f"{E}/G2_gc_onoff_newbuild/pred/e9gcoff_test_s3"
lc=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["location_classes"]
test=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]["test"]
# 학습표본 수
tr=json.load(open(f"{A}/c10_feat_train_NEW.json"))
nsam=collections.Counter(r["gt_loc"] for r in tr)
# 입력충실도 (예측혈관 우세 == GT혈관 우세)
diag={(r["case"], r["cls"]): r for r in json.load(open(f"{E}/D1_newdata/adj_diag_test.json"))}
recs=[]
for case in test:
    lp=f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz"; pp=f"{PRED}/{case}.nii.gz"
    if not (os.path.exists(lp) and os.path.exists(pp)): continue
    loc=np.asanyarray(nib.load(lp).dataobj); pr=np.asanyarray(nib.load(pp).dataobj)
    if pr.shape!=loc.shape: continue
    for c in [int(x) for x in np.unique(loc) if x]:
        lab,n=ndimage.label(loc==c)
        for i in range(1,n+1):
            m=lab==i
            if m.sum()<3: continue
            d=ndimage.binary_dilation(m, iterations=3)
            vals,cts=np.unique(pr[d], return_counts=True)
            got=0; bn=0
            for v,k in zip(vals,cts):
                if v and k>bn: got, bn = int(v), int(k)
            nm=lc[str(c)]; dg=diag.get((case,c))
            recs.append(dict(case=case, cls=c, name=nm, pred=got, ok=int(got==c),
                             detected=int(got!=0), nsam=nsam.get(nm,0),
                             input_ok=None if not dg else int(dg["gt_ves"]==dg["pred_ves"])))
json.dump(recs, open(f"{E}/D1_newdata/headroom_test.json","w"), ensure_ascii=False, indent=1)
N=len(recs)
print(f"test GT 병변 {N}건 · 검출 {sum(r['detected'] for r in recs)} · 라벨까지 정답 {sum(r['ok'] for r in recs)} ({sum(r['ok'] for r in recs)/N:.1%})\n")
def buck(n): return "1-4" if n<5 else "5-9" if n<10 else "10-19" if n<20 else "20+"
print("학습표본 수별")
print(f"{'표본':<8s} {'병변':>5s} {'검출':>6s} {'정답':>6s} {'정답률':>7s}")
g=collections.defaultdict(list)
for r in recs: g[buck(r["nsam"])].append(r)
for k in ("1-4","5-9","10-19","20+"):
    v=g[k]
    if v: print(f"{k:<8s} {len(v):5d} {sum(x['detected'] for x in v):6d} {sum(x['ok'] for x in v):6d} {sum(x['ok'] for x in v)/len(v):7.1%}")
print("\n입력충실도 × 표본수 (정답률 · 괄호는 병변수)")
print(f"{'표본':<8s} {'입력정상':>16s} {'혈관어긋남':>16s}")
for k in ("1-4","5-9","10-19","20+"):
    v=g[k]
    a=[x for x in v if x["input_ok"]==1]; b=[x for x in v if x["input_ok"]==0]
    fa=f"{sum(x['ok'] for x in a)/len(a):.0%} ({len(a)})" if a else "-"
    fb=f"{sum(x['ok'] for x in b)/len(b):.0%} ({len(b)})" if b else "-"
    print(f"{k:<8s} {fa:>16s} {fb:>16s}")
clean=[r for r in recs if r["input_ok"]==1 and r["nsam"]>=10]
miss=[r for r in clean if not r["ok"]]
print(f"\n★ '잘 맞출 수 있어야 하는' 구간 = 입력정상 ∧ 학습표본 10건+ : {len(clean)}건 · 정답 {len(clean)-len(miss)} · 틀림 {len(miss)}")
for r in miss:
    print(f"   {r['case']:<26s} GT {r['name']:<30s}(표본 {r['nsam']:2d}) → 예측 {lc[str(r['pred'])] if r['pred'] else '미검출'}")
