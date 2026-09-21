#!/usr/bin/env bash
# final_infer.py (도커 진입점) 를 test83+val42 전수에 돌려 기준 마스크와 대조 — 분류 단계 결정성 검증.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
cd $R/code/sblee/nnunet/scripts
TOPANEU_ROOT=$R /home/sblee/miniconda3/envs/sbaneu2/bin/python - <<'PYEOF'
import os,sys,numpy as np,nibabel as nib,pickle
Rt="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; sys.path.insert(0,f"{Rt}/code/sblee/nnunet/scripts")
M=pickle.load(open(f"{Rt}/code/sblee/nnunet/analysis/final_rf_seed3.pkl","rb")); tk=M["topk"]
os.environ.update({"TOPANEU_TOPK":str(tk["n"]),"TOPANEU_TOPK_VOX":str(tk["vox"]),"TOPANEU_TOPK_ICA":str(tk["ica"]),
 "TOPANEU_TOPK_MARGIN":str(tk["margin"]),"TOPANEU_TOPK_P2":str(tk["p2"]),"TOPANEU_TOPK_OR":str(tk["or"]),
 "TOPANEU_TOPK_TAU":str(tk["tau"]),"TOPANEU_TOPK_MAXN":str(tk["maxn"]),"TOPANEU_OUT_DILATE":str(M["out_dilate"])})
import c5_location_v2 as C5, d9xx_lib as L
C5.USE_POS=bool(M["use_pos"]); C5.CONF_TAU=float(M["conf_tau"]); C5.CONF_BETA_HI=float(M["conf_beta_hi"])
mdl,ves_axis,beta=M["model"],M["ves_axis"],float(M["beta"])
_,name2id=L.official_location_names(); ves_names=L.vessel_dense_names()
P=f"{Rt}/experiments/_c1_realpred"; BP=f"{Rt}/experiments/_c4_bpgraph"
_,val_ids,test_ids=L.case_ids_by_split()
tot=ok=0; bad=[]
for sp,ids,bpd in (("test",test_ids,"vespp_test"),("val",val_ids,"val_pred")):
    for c in ids:
        det=f"{P}/aneu_{sp}_P55ff/{c}.nii.gz"; vsf=f"{P}/vespp_{sp}/{c}.nii.gz"
        ref=f"{Rt}/experiments/final_pred_seed3_{sp}/{c}.nii.gz"
        if not (os.path.exists(det) and os.path.exists(ref)): continue
        vi=nib.load(vsf); ves=np.asanyarray(vi.dataobj); spc=np.array(vi.header.get_zooms()[:3],dtype=float)
        ai=nib.load(det); pr=np.asanyarray(ai.dataobj)
        nodes=C5.load_bp(f"{BP}/{bpd}", c)
        rows,les=C5.extract_case_rows(pr,ves,spc,ves_names,nodes,None)
        for r in rows: r["case"]=c
        out=np.zeros(pr.shape,dtype=np.int32)
        for r in rows:
            nm=C5.predict_one(mdl,r,beta)
            if nm is None: continue
            oid=name2id.get(nm)
            if oid is None: continue
            out[les==r["lesion_mask_idx"]]=oid
            if C5.TOPK_N>1: C5._emit_topk(out,les,r,mdl,beta,nm,name2id)
        b=np.asanyarray(nib.load(ref).dataobj); tot+=1
        if out.shape==b.shape and np.array_equal(out.astype(np.uint8),b.astype(np.uint8)): ok+=1
        else: bad.append((sp,c,int((out!=b).sum()) if out.shape==b.shape else "shape"))
print(f"\n=== final_infer 경로 전수 대조 (분류 단계 결정성) ===")
print(f"  케이스 {tot} · 기준과 완전일치 {ok} · 불일치 {len(bad)}")
for x in bad[:10]: print("    ★",x)
PYEOF
echo "VERIFYALL_DONE $(TZ=Asia/Seoul date +%H:%M) KST"
