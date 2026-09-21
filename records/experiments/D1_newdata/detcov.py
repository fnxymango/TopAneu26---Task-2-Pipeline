#!/usr/bin/env python3
"""detcov.py — 검출층만 본다. GT 병변이 검출 마스크에 잡히는가를 통합본(b1ff) vs 우리(e9ff) 로 비교.

분류기는 검출이 덮은 병변에만 손댈 수 있으므로, 여기서 놓친 것은 분류 개선으로 되찾을 수 없다.
판정 규약은 perclass.py 와 같다 — GT 병변을 3회 팽창시킨 영역이 검출 마스크와 겹치면 검출.
"""
import json, os, collections
import numpy as np, nibabel as nib
from scipy import ndimage
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; E=f"{R}/experiments"; P=f"{E}/_c1_realpred"
S=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json")); LC=S["location_classes"]; SP=S["splits"]
NAME={int(k):v for k,v in LC.items()}
DETS=["b1ff","e9ff"]
recs=[]
for sp in ("test","val"):
    for case in SP[sp]:
        gp=f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz"
        dps={d:f"{P}/aneu_{sp}_{d}/{case}.nii.gz" for d in DETS}
        if not os.path.exists(gp) or not all(os.path.exists(v) for v in dps.values()): continue
        gi=nib.load(gp); gt=np.asanyarray(gi.dataobj); vx=float(np.prod(gi.header.get_zooms()[:3]))
        dm={d:np.asanyarray(nib.load(v).dataobj) for d,v in dps.items()}
        for c in sorted({int(x) for x in np.unique(gt) if x}):
            lab,n=ndimage.label(gt==c)
            for i in range(1,n+1):
                m=lab==i
                if m.sum()<3: continue
                d3=ndimage.binary_dilation(m, iterations=3)
                recs.append(dict(split=sp, case=case, cls=c, name=NAME[c],
                    vox=int(m.sum()), dia=float(2*(3*(m.sum()*vx)/(4*np.pi))**(1/3)),
                    **{f"det_{k}": int(bool((d3&(v>0)).sum())) for k,v in dm.items()}))
json.dump(recs, open(f"{E}/D1_newdata/detcov.json","w"), ensure_ascii=False, indent=1)
print(f"저장 {len(recs)}병변")
