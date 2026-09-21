#!/usr/bin/env bash
# 재현 검증: 원본 영상 → (번들 구성요소만으로) 전 단계 → 기준 출력과 복셀 대조 (2026-08-28).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
FORK=/home/sblee/miniconda3/envs/sbaneu2/bin        # nnunetv2 2.5 = vendor 포크
STOCK=/home/sblee/miniconda3/envs/sblee_topaneu/bin # nnunetv2 2.8.1 stock (검출기 원본 환경)
W=$E/verify_e2e2; rm -rf $W; mkdir -p $W/{in722,in_raw,det_fork,det_stock,ves,vespp,bp,out}
export TOPANEU_ROOT=$R nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
cd $S
CASES="topaneu_center2_ct_192 topaneu_center1_mr_148 topaneu_center2_mr_038"
log(){ echo "[verify $(TZ=Asia/Seoul date +%H:%M)] $*"; }

log "STEP0 robust-z (p_build_3cls 와 동일 함수)"
$FORK/python - $W "$CASES" <<'PYEOF'
import sys, os, importlib.util, numpy as np, nibabel as nib
W, cases = sys.argv[1], sys.argv[2].split()
spec=importlib.util.spec_from_file_location("pb","/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/scripts/p_build_3cls.py")
pb=importlib.util.module_from_spec(spec); spec.loader.exec_module(pb)
D="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/dataset/TopAneu/images"
for c in cases:
    img=nib.load(f"{D}/{c}_0000.nii.gz"); arr=np.asanyarray(img.dataobj)
    norm,_,_,_=pb.robust_z(arr, pb.modality(c))
    nib.save(nib.Nifti1Image(norm.astype(np.float32), img.affine, img.header), f"{W}/in722/{c}_0000.nii.gz")
    os.system(f"cp -L {D}/{c}_0000.nii.gz {W}/in_raw/")
    ref=f"/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/_c1_realpred/in_test_722/{c}_0000.nii.gz"
    a=np.asanyarray(nib.load(f"{W}/in722/{c}_0000.nii.gz").dataobj); b=np.asanyarray(nib.load(ref).dataobj)
    print(f"  [{c}] 정규화 입력 vs 기준 in_test_722: 최대차 {np.abs(a-b).max():.3e}")
PYEOF

DRES=$E/P1_pjh3cls_stock250_iso04_f0/results
log "STEP1a 검출 5폴드 — 포크(2.5)"
nnUNet_results=$DRES CUDA_VISIBLE_DEVICES=0 $FORK/nnUNetv2_predict -i $W/in722 -o $W/det_fork \
  -d 722 -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
  --disable_tta -npp 2 -nps 2 > $W/det_fork.log 2>&1 || log "  포크 검출 실패"
log "STEP1b 검출 5폴드 — stock(2.8.1, 원본 환경)"
nnUNet_results=$DRES CUDA_VISIBLE_DEVICES=1 $STOCK/nnUNetv2_predict -i $W/in722 -o $W/det_stock \
  -d 722 -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
  --disable_tta -npp 2 -nps 2 > $W/det_stock.log 2>&1 || log "  stock 검출 실패"

log "STEP2 혈관 fold0 (포크 필수) — chk=best"
nnUNet_results=$E/V4-2_vessel_classweighted_417_500ep/results CUDA_VISIBLE_DEVICES=0 \
  $FORK/nnUNetv2_predict -i $W/in_raw -o $W/ves -d 800 -c 3d_fullres -f 0 \
  -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep -p nnUNetResEncUNetMPlans \
  -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 > $W/ves.log 2>&1 || log "  혈관 실패"
log "STEP3 혈관 후처리"
$FORK/python postprocess_vessel.py apply $W/ves $W/vespp > $W/vespp.log 2>&1 || log "  후처리 실패"
log "STEP4 분기점"
$FORK/python c4_branchpoint_graph.py --vessel-dir $W/vespp --out $W/bp > $W/bp.log 2>&1 || log "  분기점 실패"

log "STEP5 검출필터(GT불요) + 최종추론 + 대조"
$FORK/python - $W "$CASES" <<'PYEOF'
import sys, os, json, subprocess, numpy as np, nibabel as nib
sys.path.insert(0,"/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/scripts")
from det_filter import filter_case
W, cases = sys.argv[1], sys.argv[2].split()
Rt="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; P=f"{Rt}/experiments/_c1_realpred"; BP=f"{Rt}/experiments/_c4_bpgraph"
def cmp(a,b,name):
    if not os.path.exists(b): print(f"    {name:22s} 기준 없음"); return
    x=np.asanyarray(nib.load(a).dataobj); y=np.asanyarray(nib.load(b).dataobj)
    if x.shape!=y.shape: print(f"    {name:22s} ★shape {x.shape} vs {y.shape}"); return
    d=int((x!=y).sum()); print(f"    {name:22s} {'일치' if d==0 else f'★불일치 {d}복셀 ({d/x.size:.2e})'}")
print("\n=== 단계별 대조 ===")
for c in cases:
    print(f"  [{c}]")
    for tag,src in (("검출(포크)",f"{W}/det_fork/{c}.nii.gz"),("검출(stock)",f"{W}/det_stock/{c}.nii.gz")):
        if not os.path.exists(src): print(f"    {tag:22s} 없음"); continue
    # 두 검출 출력끼리 비교
    fa,fb=f"{W}/det_fork/{c}.nii.gz", f"{W}/det_stock/{c}.nii.gz"
    if os.path.exists(fa) and os.path.exists(fb):
        x=np.asanyarray(nib.load(fa).dataobj); y=np.asanyarray(nib.load(fb).dataobj)
        d=int((x!=y).sum()); print(f"    {'포크 vs stock 검출':22s} {'일치' if d==0 else f'★불일치 {d}복셀'}")
    cmp(f"{W}/vespp/{c}.nii.gz", f"{P}/vespp_test/{c}.nii.gz", "혈관후처리 vs 기준")
    # 검출 이진화 + 필터 (stock 우선, 없으면 포크)
    det = fb if os.path.exists(fb) else fa
    if os.path.exists(det) and os.path.exists(f"{W}/vespp/{c}.nii.gz"):
        di=nib.load(det); an=(np.asanyarray(di.dataobj)==2)
        vi=nib.load(f"{W}/vespp/{c}.nii.gz"); ves=np.asanyarray(vi.dataobj)
        sp=np.array(di.header.get_zooms()[:3],dtype=float)
        m=filter_case(an,ves,sp,5,1.0)
        nib.save(nib.Nifti1Image(m,di.affine,di.header), f"{W}/out/{c}_det.nii.gz")
        cmp(f"{W}/out/{c}_det.nii.gz", f"{P}/aneu_test_P55ff/{c}.nii.gz", "검출필터 vs 기준")
        # 분기점 json 비교
        ja,jb=f"{W}/bp/{c}.json", f"{BP}/vespp_test/{c}.json"
        if os.path.exists(ja) and os.path.exists(jb):
            na=json.load(open(ja)).get("nodes",[]); nb=json.load(open(jb)).get("nodes",[])
            print(f"    {'분기점 노드수':22s} {len(na)} vs {len(nb)} {'일치' if na==nb else '★내용차이'}")
        # 최종 추론
        r=subprocess.run([sys.executable,"final_infer.py","--model",f"{Rt}/code/sblee/nnunet/analysis/final_rf_seed3.pkl",
            "--aneu",f"{W}/out/{c}_det.nii.gz","--vessel",f"{W}/vespp/{c}.nii.gz","--bp",f"{W}/bp/{c}.json",
            "--out",f"{W}/out/{c}_loc.nii.gz"],cwd=f"{Rt}/code/sblee/nnunet/scripts",capture_output=True,text=True,
            env={**os.environ,"TOPANEU_ROOT":Rt})
        if os.path.exists(f"{W}/out/{c}_loc.nii.gz"):
            cmp(f"{W}/out/{c}_loc.nii.gz", f"{Rt}/experiments/final_pred_seed3_test/{c}.nii.gz", "★최종 위치라벨 vs 기준")
        else:
            print("    최종추론 실패:", r.stderr.strip().splitlines()[-1] if r.stderr else "?")
PYEOF
echo "VERIFY_DONE $(TZ=Asia/Seoul date +%H:%M) KST"
