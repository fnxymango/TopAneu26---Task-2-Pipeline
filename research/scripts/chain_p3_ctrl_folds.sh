#!/usr/bin/env bash
# (1) stock fold0 단독 대조군 — ResEncL 효과를 fold0 운에서 분리한다 (2026-08-28).
#     지금까지 비교하던 X5 는 5폴드 앙상블이라 P3 fold0 와 조건이 다르다.
# (2) 이어서 P3 fold3, fold4 학습 (게이트 PASS 로 5폴드 확대 승인됨).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; E=$R/experiments; P=$E/_c1_realpred
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
cd $S; ST=$E/p3_status.log
log(){ echo "[P3-ctrl $(TZ=Asia/Seoul date +'%m-%d %H:%M')] $*" | tee -a "$ST"; }
echo $$ > $E/p3_ctrl.pid

# ---- (1) 대조군: stock fold0 ----
log "대조군 stock fold0 val 추론 (GPU0)"
nnUNet_results=$E/P1_pjh3cls_stock250_iso04_f0/results CUDA_VISIBLE_DEVICES=0 \
  "$ENVBIN/nnUNetv2_predict" -i "$P/in_val_722" -o "$P/aneu_val_stockf0" \
  -d 722 -c 3d_fullres -f 0 -p nnUNetPlans -tr nnUNetTrainer_250epochs \
  -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 > "$E/p3ctrl_val_pred.log" 2>&1 \
  || { log "★대조군 추론 실패"; }
$PY - "$P/aneu_val_stockf0" "$P/aneu_val_stockf0_bin" <<'PYEOF'
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
$PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_val_stockf0_bin" --vessel-dir "$P/vespp_val" \
    --split val --tag stockf0_val > "$E/p3ctrl_val_c7.txt" 2>&1 || log "★대조군 c7 실패"
$PY - <<'PYEOF' | tee -a "$ST"
import re
def best(p):
    b=None
    for L in open(p,errors="replace"):
        m=re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\((\d+)/(\d+)\)\s+([\d.]+)\s+(\d+)",L)
        if not m: continue
        s,fp=float(m.group(3)),float(m.group(6))
        if b is None or (s,-fp)>(b[0],-b[1]): b=(s,fp,m.group(4),m.group(5))
    return b
E="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/"
c=best(E+"p3ctrl_val_c7.txt"); p=best(E+"p3_val_c7.txt")
print("\n=== 아키텍처 효과 분리 (val 42, 같은 fold0 · 같은 pjh 레시피) ===")
if c: print(f"  stock  fold0 : 민감도 {c[0]:.3f} ({c[2]}/{c[3]}) · FP/case {c[1]:.2f}")
if p: print(f"  ResEncL fold0 : 민감도 {p[0]:.3f} ({p[2]}/{p[3]}) · FP/case {p[1]:.2f}")
if c and p: print(f"  Δ            : 민감도 {p[0]-c[0]:+.3f} ({int(p[2])-int(c[2]):+d}병변) · FP/case {p[1]-c[1]:+.2f}")
print("  (참고) X5 5폴드 앙상블 : 민감도 0.837 (36/43) · FP/case 0.17 — cfg 5/1.0 실측. 이전 0.860/0.43 은 오기, X5_BASELINE.md 참조")
PYEOF

# ---- (2) P3 fold3, fold4 ----
EXP=P3_pjh3cls_resencl_iso04_f0; PLANS=nnUNetResEncUNetLPlans722iso04; TR=nnUNetTrainer_250epochs
export nnUNet_results=$E/$EXP/results
D="$nnUNet_results/Dataset722_TopAneuPjh3cls417/${TR}__${PLANS}__3d_fullres"
for f in 3 4; do
  [ -f "$D/fold_${f}/checkpoint_final.pth" ] && { log "fold${f} 이미 완료 — 건너뜀"; continue; }
  log "fold${f} 학습 시작 (GPU0)"
  CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_train" 722 3d_fullres $f -p "$PLANS" -tr "$TR" \
    > "$E/p3_train_f${f}.log" 2>&1
  [ -f "$D/fold_${f}/checkpoint_best.pth" ] && log "fold${f} 학습 완료" || { log "★fold${f} 학습 실패"; exit 1; }
done
log "GPU0 체인 종료 (대조군 + fold3 + fold4)"
