#!/usr/bin/env python3
"""클래스별 실패 진단 — test+val 130병변. 각 병변에 실패 지점과 기전을 태깅한다."""
import json, os, collections, numpy as np, nibabel as nib
from scipy import ndimage
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; E=f"{R}/experiments"; P=f"{E}/_c1_realpred"
S=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
lc=S["location_classes"]; splits=S["splits"]
vlab=json.load(open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"]
vinv={int(v):k for k,v in vlab.items() if int(v)}
tr=json.load(open(f"{R}/code/sblee/nnunet/analysis/c10_feat_train_NEW.json"))
nsam=collections.Counter(r["gt_loc"] for r in tr)
mv=collections.defaultdict(collections.Counter)
for r in tr:
    ov={k:v for k,v in r["overlap"].items() if v}
    if ov: mv[r["gt_loc"]][max(ov,key=ov.get)]+=1
modal={k:c.most_common(1)[0][0] for k,c in mv.items()}
def dom(msk, ves):
    vals,cts=np.unique(ves[msk], return_counts=True); b=None; bn=0
    for v,n in zip(vals,cts):
        if v and n>bn: b,bn=int(v),int(n)
    return vinv.get(b) if b else None
recs=[]
for sp in ("test","val"):
    PR=f"{E}/G2_gc_onoff_newbuild/pred/e9gcoff_{sp}_s3"
    for case in splits[sp]:
        f=lambda p: os.path.exists(p)
        lp=f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz"; pp=f"{PR}/{case}.nii.gz"
        gvp=f"{R}/dataset/TopAneu/vessel_masks/{case}.nii.gz"; pvp=f"{P}/vespp_{sp}/{case}.nii.gz"
        rawp=f"{P}/aneu_{sp}_e9_bin/{case}.nii.gz"; ffp=f"{P}/aneu_{sp}_e9ff/{case}.nii.gz"
        if not all(os.path.exists(x) for x in (lp,pp,gvp,pvp,rawp,ffp)): continue
        im=nib.load(lp); loc=np.asanyarray(im.dataobj); vx=np.prod(im.header.get_zooms()[:3])
        cls=[int(x) for x in np.unique(loc) if x]
        if not cls: continue
        pr=np.asanyarray(nib.load(pp).dataobj); gv=np.asanyarray(nib.load(gvp).dataobj)
        pv=np.asanyarray(nib.load(pvp).dataobj); raw=np.asanyarray(nib.load(rawp).dataobj); ff=np.asanyarray(nib.load(ffp).dataobj)
        if pr.shape!=loc.shape: continue
        for c in cls:
            lab,n=ndimage.label(loc==c)
            for i in range(1,n+1):
                m=lab==i
                if m.sum()<3: continue
                d=ndimage.binary_dilation(m, iterations=3)
                vals,cts=np.unique(pr[d],return_counts=True); got=0;bn=0
                for v,k in zip(vals,cts):
                    if v and k>bn: got,bn=int(v),int(k)
                nm=lc[str(c)]
                recs.append(dict(split=sp, case=case, cls=c, name=nm,
                    dia=float(2*(3*(m.sum()*vx)/(4*np.pi))**(1/3)),
                    pred=got, ok=int(got==c),
                    raw=int((d&(raw>0)).sum()), ff=int((d&(ff>0)).sum()),
                    gt_ves=dom(d,gv), pred_ves=dom(d,pv), modal=modal.get(nm), nsam=nsam.get(nm,0)))
json.dump(recs, open(f"{E}/D1_newdata/perclass.json","w"), ensure_ascii=False, indent=1)
print(f"저장 {len(recs)}병변 (test {sum(r['split']=='test' for r in recs)} · val {sum(r['split']=='val' for r in recs)})")
