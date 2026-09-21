#!/bin/bash
# X5 검출기 + gC 재판정 (2026-08-26). 규칙: test·val 둘 다 T16+gC 초과여야 교체.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
export TOPANEU_ROOT=$R nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
cd $S
st(){ echo "[STEP] $(TZ=Asia/Seoul date +%H:%M) $*"; }

gc_eval(){ # split aneudir seed tag
  local sp=$1 an=$2 sd=$3 tg=$4 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$an" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tg}_s${sd}" > "$E/${tg}_s${sd}.log" 2>&1
}

# (1) test — 예측이 이미 있으므로 즉시 5시드 병렬
st "X5+gC test 5시드 시작 (aneu_test_P55ff)"
for SD in 0 1 2 3 4; do gc_eval test aneu_test_P55ff $SD x5g_test & done

# (2) val — GPU 추론 (5폴드 확률평균) -> 라벨2 -> c7(force-cfg 5,1.0)
OUT=$P/aneu_val_P55f; BIN=$P/aneu_val_P55f_bin; FF=$P/aneu_val_P55ff
if [ "$(ls "$FF"/*.nii.gz 2>/dev/null | wc -l)" -lt 42 ]; then
  st "X5 val 5폴드 추론 시작 (GPU0)"
  nnUNet_results="$E/P1_pjh3cls_stock250_iso04_f0/results" CUDA_VISIBLE_DEVICES=0 \
    "$ENVBIN/nnUNetv2_predict" -i "$P/in_val_722" -o "$OUT" -d 722 -c 3d_fullres \
    -f 0 1 2 3 4 -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
    --disable_tta -npp 2 -nps 2 > "$E/x5g_val_pred.log" 2>&1 || { echo FAIL_pred; exit 1; }
  $PY - "$OUT" "$BIN" <<'PYEOF' || { echo FAIL_bin; exit 1; }
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True)
n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  st "c7 필터 (test 와 동일 cfg 5,1.0)"
  $PY -u c7_detect_postproc.py --aneu-dir "$BIN" --vessel-dir "$P/vespp_val" \
      --split val --tag x5g_val --save-best "$FF" --force-cfg "5,1.0" \
      > "$E/x5g_val_c7.txt" 2>&1 || { echo FAIL_c7; exit 1; }
fi
st "X5+gC val 5시드 시작"
for SD in 0 1 2 3 4; do gc_eval val aneu_val_P55ff $SD x5g_val & done
wait
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
