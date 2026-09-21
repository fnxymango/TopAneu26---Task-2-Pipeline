#!/usr/bin/env bash
# P3 5폴드 e2e 판정 (2026-08-29).
# 단일 레버: 검출기만 X5(stock 5폴드) -> P3(ResEncL 5폴드) 로 교체. 나머지 전부 동결.
#   c7 는 X5 와 동일한 프로덕션 설정 force-cfg 5,1.0 을 쓴다.
#   (fold0 스윕 최적은 0,2.0 이었지만 val 로 재튜닝하면 val 로 평가할 수 없다 — 누수)
# 채택 규칙: test 83 AND val 42 e2e Δ 둘 다 양수일 때만.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
PLANS=nnUNetResEncUNetLPlans722iso04; TR=nnUNetTrainer_250epochs
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
export nnUNet_results=$E/P3_pjh3cls_resencl_iso04_f0/results
cd $S; ST=$E/p3_status.log
log(){ echo "[P3-e2e $(TZ=Asia/Seoul date +'%m-%d %H:%M')] $*" | tee -a "$ST"; }
bin2(){ $PY - "$1" "$2" <<'PYEOF'
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
}
# --- 추론: val 은 GPU0, test 는 GPU1 동시 ---
log "5폴드 추론 시작 — val(GPU0) · test(GPU1) 병렬"
( CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" -i "$P/in_val_722" -o "$P/aneu_val_p3x5" \
    -d 722 -c 3d_fullres -f 0 1 2 3 4 -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth \
    --disable_tta -npp 2 -nps 2 > "$E/p3e2e_val_pred.log" 2>&1 ) &
V=$!
( CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" -i "$P/in_test_722" -o "$P/aneu_test_p3x5" \
    -d 722 -c 3d_fullres -f 0 1 2 3 4 -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth \
    --disable_tta -npp 2 -nps 2 > "$E/p3e2e_test_pred.log" 2>&1 ) &
T=$!
wait $V || { log "★val 추론 실패"; exit 1; }
wait $T || { log "★test 추론 실패"; exit 1; }
log "추론 완료 · val $(ls $P/aneu_val_p3x5/*.nii.gz|wc -l) · test $(ls $P/aneu_test_p3x5/*.nii.gz|wc -l)"

for sp in val test; do
  bin2 "$P/aneu_${sp}_p3x5" "$P/aneu_${sp}_p3x5_bin"
  $PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_p3x5_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag p3x5_${sp} --save-best "$P/aneu_${sp}_p3x5ff" --force-cfg "5,1.0" \
      > "$E/p3e2e_${sp}_c7.txt" 2>&1 || { log "★${sp} c7 실패"; exit 1; }
done
log "c7 완료 — e2e 5시드 (gC 위에서)"

one(){ local sp=$1 sd=$2 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_p3x5ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "p3x5_${sp}_s${sd}" > "$E/p3x5_${sp}_s${sd}.log" 2>&1
}
for sd in 0 1 2 3 4; do one test $sd & one val $sd & done
wait
$PY - <<'PYEOF' | tee -a "$ST"
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
X5={"test":"c5_eval_test_x5g_test_s%d.json","val":"c5_eval_val_x5g_val_s%d.json"}
print("\n=== P3(ResEncL 5폴드) + gC  vs  X5(stock 5폴드) + gC ===")
ok=True
for sp,n in (("test",83),("val",42)):
    fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_p3x5_{sp}_s*.json"))
    if not fs: print(f"  {sp}: 결과 없음"); ok=False; continue
    sds=[int(f.split("_s")[-1][:-5]) for f in fs]
    print(f"  [{sp} {n}]")
    for key,lab in (("adjusted_div_present","covered_gt"),("official_div52","official/52")):
        x=np.array([json.load(open(f))[key]["MCC"] for f in fs])
        g=np.array([json.load(open(f"{A}/"+X5[sp]%s))[key]["MCC"] for s in sds])
        d=x-g
        print(f"    {lab:11s} P3 {x.mean():.4f}±{x.std(ddof=1):.4f}  X5 {g.mean():.4f}  Δ{d.mean():+.4f}  양수 {int((d>0).sum())}/{len(d)}")
        if lab=="covered_gt" and d.mean()<=0: ok=False
    j=json.load(open(fs[0])); k=json.load(open(f"{A}/"+X5[sp]%sds[0]))
    print(f"    blob {j['n_lesions_predicted']} vs {k['n_lesions_predicted']}")
print(f"\n판정: {'채택 (두 집합 covered_gt Δ 양수)' if ok else '기각'}")
PYEOF
log "P3_E2E_DONE"
