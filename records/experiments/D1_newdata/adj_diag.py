#!/usr/bin/env python3
"""인접분절 혼동의 원인 분해 — test 83.
각 GT 병변에 대해
  (a) GT 라벨이 학습 관례(클래스→우세혈관 최빈값)와 맞는가   → 어긋나면 '주석 불일치' (고칠 수 없음)
  (b) 예측혈관의 우세혈관이 GT혈관의 우세혈관과 같은가        → 다르면 '상류 혈관경계 오류' (고칠 수 있음)
"""
import json, os, collections, numpy as np, nibabel as nib
from scipy import ndimage
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
A=f"{R}/code/sblee/nnunet/analysis"; P=f"{R}/experiments/_c1_realpred"
lc=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["location_classes"]
split=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]["test"]
vlab=json.load(open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"]
vinv={int(v):k for k,v in vlab.items() if int(v)}

# 1) 학습 271병변에서 클래스 → 우세혈관 최빈값 (관례)
rows=json.load(open(f"{A}/c10_feat_train_NEW.json"))
byc=collections.defaultdict(collections.Counter)
for r in rows:
    ov={k:v for k,v in r["overlap"].items() if v}
    if ov: byc[r["gt_loc"]][max(ov,key=ov.get)]+=1
modal={c:cnt.most_common(1)[0][0] for c,cnt in byc.items()}
name2id={v:int(k) for k,v in lc.items()}

def dominant(mask, ves):
    d=ndimage.binary_dilation(mask, iterations=3)
    vals,cts=np.unique(ves[d], return_counts=True)
    best=None; bn=0
    for v,n in zip(vals,cts):
        if v and n>bn: best, bn = int(v), int(n)
    return (vinv.get(best,str(best)) if best else None), bn

out=[]; skipped=[]
for case in split:
    lp=f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz"
    gp=f"{R}/dataset/TopAneu/vessel_masks/{case}.nii.gz"
    pp=f"{P}/vespp_test/{case}.nii.gz"
    if not all(os.path.exists(x) for x in (lp,gp,pp)): skipped.append(case); continue
    loc=np.asanyarray(nib.load(lp).dataobj)
    cls=[int(x) for x in np.unique(loc) if x]
    if not cls: continue
    gv=np.asanyarray(nib.load(gp).dataobj); pv=np.asanyarray(nib.load(pp).dataobj)
    if pv.shape!=loc.shape: skipped.append(case+"(shape)"); continue
    for c in cls:
        lab,n=ndimage.label(loc==c)
        for i in range(1,n+1):
            m=lab==i
            if m.sum()<3: continue
            dg,_=dominant(m,gv); dp,_=dominant(m,pv)
            out.append(dict(case=case, cls=c, name=lc[str(c)], gt_ves=dg, pred_ves=dp,
                            modal=modal.get(lc[str(c)]), nvox=int(m.sum())))
json.dump(out, open(f"{R}/experiments/D1_newdata/adj_diag_test.json","w"), ensure_ascii=False, indent=1)
print(f"test GT 병변 {len(out)}건 분석 (건너뜀 {len(skipped)})\n")

def report(sel, title):
    if not sel: return
    ann_bad=[r for r in sel if r["modal"] and r["gt_ves"]!=r["modal"]]
    ok=[r for r in sel if not (r["modal"] and r["gt_ves"]!=r["modal"])]
    up_bad=[r for r in ok if r["gt_ves"]!=r["pred_ves"]]
    clean=[r for r in ok if r["gt_ves"]==r["pred_ves"]]
    print(f"[{title}] 병변 {len(sel)}건")
    print(f"   주석 불일치 (GT라벨 ↔ GT혈관 관례 어긋남)   {len(ann_bad):3d}  {len(ann_bad)/len(sel):5.1%}   ← 모델로 못 고침")
    print(f"   상류 혈관경계 오류 (예측혈관 ≠ GT혈관)      {len(up_bad):3d}  {len(up_bad)/len(sel):5.1%}   ← 혈관 개선으로 고칠 몫")
    print(f"   입력 정상                                  {len(clean):3d}  {len(clean)/len(sel):5.1%}")
    for r in ann_bad[:6]: print(f"      [주석] {r['case']:<26s} {r['name']:<30s} GT혈관 {r['gt_ves']}  (관례 {r['modal']})")
    for r in up_bad[:6]:  print(f"      [상류] {r['case']:<26s} {r['name']:<30s} GT혈관 {r['gt_ves']} → 예측 {r['pred_ves']}")
    print()
report(out, "test 전체")
report([r for r in out if "M1" in r["name"] or "M1-M2" in r["name"]], "M1 계열")
