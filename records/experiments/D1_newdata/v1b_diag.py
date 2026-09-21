"""V1-B 사후 진단 — test/val 의 1.9 BA-SCA GT 병변이 추론 입력(예측 혈관·검출 blob)에서 어떻게 보이나."""
import json, os, sys
import numpy as np, nibabel as nib
from scipy import ndimage
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; P=f"{R}/experiments/_c1_realpred"; BP=f"{R}/experiments/_c4_bpgraph"
sys.path.insert(0,f"{R}/code/sblee/nnunet/scripts"); os.environ.setdefault("TOPANEU_ROOT",R)
import c5_v1b as C
S=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json")); LOC={int(k):v for k,v in S["location_classes"].items()}
vn=C.L.vessel_dense_names()
tgt={k for k,v in LOC.items() if "1.9 BA-SCA" in v}
for sp in ("test","val"):
    for c in S["splits"][sp]:
        g=np.asanyarray(nib.load(f"{R}/dataset/TopAneu/location_masks/{c}.nii.gz").dataobj)
        ks=[int(k) for k in np.unique(g) if int(k) in tgt]
        if not ks: continue
        vi=nib.load(f"{P}/vespp_{sp}/{c}.nii.gz"); ves=np.asanyarray(vi.dataobj); spc=np.array(vi.header.get_zooms()[:3],float)
        gv=np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{c}.nii.gz").dataobj)
        an=np.asanyarray(nib.load(f"{P}/aneu_{sp}_b1ff/{c}.nii.gz").dataobj)
        bpd=f"{BP}/{'vespp_test' if sp=='test' else 'val_pred'}"
        nodes=C.load_bp(bpd,c)
        rows,les=C.extract_case_rows(an,ves,spc,vn,nodes)
        for k in ks:
            gm=g==k
            print(f"\n{sp} {c} GT {LOC[k]} ({int(gm.sum())}vox)")
            for s in ("R","L"):
                print(f"   GT혈관에 {s}-SCA 존재 {int((gv==C.VID if False else 0) if False else (gv==[i for i,n in vn.items() if n==f'{s}-SCA'][0]).sum())}vox · 예측혈관 {int((ves==[i for i,n in vn.items() if n==f'{s}-SCA'][0]).sum())}vox")
            hit=False
            for r in rows:
                m=les==r["lesion_mask_idx"]
                if not (m&ndimage.binary_dilation(gm,iterations=3)).any(): continue
                hit=True
                dm=r.get("dist_mm") or {}; bp=r.get("bp_mm") or []
                print(f"   검출blob {int(m.sum())}vox · SCA 거리 R {dm.get('R-SCA')} L {dm.get('L-SCA')} · BA {dm.get('BA')} · "
                      f"BA-SCA노드 R {bp[C._V1B_NODE_IDX['R']]} L {bp[C._V1B_NODE_IDX['L']]} · P1P2 R {dm.get('R-P1P2')} L {dm.get('L-P1P2')}")
            if not hit: print("   ★검출 blob 없음")
