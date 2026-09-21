#!/usr/bin/env bash
# A7 test 추론 + 현행 검출기와 병변 겹침 (2026-08-18).
# val 에서는 A7 과 현행 5-fold 가 **정확히 같은 33개**를 찾고 같은 10개를 놓쳤다.
# test 에서도 같은지 확인한다 — 다르면 합집합으로 민감도를 살 여지가 생긴다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"; P="$E/_c1_realpred"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
AEXP="A7_tversky_a15b85_417_f0"
log(){ echo "[a7t $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

if [ "$(ls "$P/aneu_test_a7"/*.nii.gz 2>/dev/null | wc -l)" -lt 80 ]; then
  log "GPU: A7 test 추론"
  nnUNet_results="$E/$AEXP/results" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
    -i "$P/in_test" -o "$P/aneu_test_a7" -d 720 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTverskyTopkCE_a15b85 -p nnUNetResEncUNetLPlansAdaptive \
    -chk checkpoint_best.pth --disable_tta --continue_prediction -npp 2 -nps 2 \
    > "$E/a7_pred_test.log" 2>&1 || log "  추론 실패"
fi
log "test 마스크 $(ls "$P/aneu_test_a7"/*.nii.gz 2>/dev/null | wc -l)/83"

log "=== c7 필터 스윕 (test) ==="
$PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_test_a7" --vessel-dir "$P/vespp_test" \
   --split test --tag a7_test 2>&1 | tail -14 || log "  실패"

log "=== 현행 5-fold 와 병변 겹침 (test) ==="
$PY -u - <<'PYEOF'
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L
ST=np.ones((3,3,3),bool)
LAB=L.TOPANEU_ROOT/"nnunet"/"nnUNet_raw"/"Dataset720_TopAneuBinary417"/"labelsTr"
P=L.TOPANEU_ROOT/"experiments"/"_c1_realpred"
ids=L.case_ids_by_split()[2]
def filt(pr,ves,sp,mv,md):
    lab,n=ndi.label(pr,structure=ST)
    dist=ndi.distance_transform_edt(~(ves>0),sampling=sp) if (ves>0).any() else np.full(pr.shape,1e9)
    keep=np.zeros(n+1,bool)
    for l in range(1,n+1):
        s=lab==l
        if s.sum()>=mv and dist[s].min()<=md: keep[l]=True
    return keep[lab]
A={}; FP={}; tot=set()
for cid in ids:
    gp=LAB/f"{cid}.nii.gz"
    if not gp.exists(): continue
    gt=np.asanyarray(nib.load(gp).dataobj)>0
    glab,gn=ndi.label(gt,structure=ST); tot|={(cid,i) for i in range(1,gn+1)}
    vp=P/"vespp_test"/f"{cid}.nii.gz"
    ves=np.asanyarray(nib.load(vp).dataobj) if vp.exists() else np.zeros_like(gt,np.int16)
    sp=np.array(nib.load(gp).header.get_zooms()[:3],float)
    for name,d,mv,md in (("prod",P/"aneu_test_probavgf",0,1e9),("a7",P/"aneu_test_a7",40,1.0)):
        pp=d/f"{cid}.nii.gz"
        if not pp.exists(): continue
        pr=np.asanyarray(nib.load(pp).dataobj)>0
        if mv: pr=filt(pr,ves,sp,mv,md)
        plab,pn=ndi.label(pr,structure=ST); f=set(); fp=0
        for l in range(1,pn+1):
            s=plab==l; g=set(int(x) for x in np.unique(glab[s]) if x>0)
            if g: f|={(cid,x) for x in g}
            else: fp+=1
        A.setdefault(name,set()).update(f); FP[name]=FP.get(name,0)+fp
p,a=A.get("prod",set()),A.get("a7",set())
print(f"\nGT 병변 {len(tot)} (test 83케이스)")
print(f"현행 5-fold  {len(p)} (FP {FP.get('prod',0)})")
print(f"A7 fold0     {len(a)} (FP {FP.get('a7',0)})")
print(f"둘다 찾음    {len(p&a)}")
print(f"현행만       {len(p-a)}")
print(f"A7만         {len(a-p)}   <- 합집합 이득")
print(f"둘다 놓침    {len(tot-p-a)}")
print(f"합집합       {len(p|a)}/{len(tot)} = {len(p|a)/len(tot):.3f}  (현행 {len(p)/len(tot):.3f})")
PYEOF
log "=== 완료 ==="
