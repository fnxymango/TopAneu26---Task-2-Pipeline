#!/bin/bash
# F2 검출기 확률 내보내기 (2026-08-26 밤): P55 앙상블 --save_probabilities 재추론(test+val)
#  -> blob별 검출확신 -> 환각 분리 AUC 진단. 판정 없음(진단만) — 게이트化는 내일 two-set e2e 로.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; E=$R/experiments; P=$E/_c1_realpred
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
export TOPANEU_ROOT=$R nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
ST=$E/chain_status.md
# UNION 까지 끝나 GPU 경쟁 없을 때 시작
until grep -q "UNION_DONE" "$ST" 2>/dev/null; do sleep 60; done
echo "[$(TZ=Asia/Seoul date +%H:%M)] F2 확률 내보내기 시작 (GPU0)" >> "$ST"
RES="$E/P1_pjh3cls_stock250_iso04_f0/results"
for sp in test val; do
  OUT="$P/aneu_${sp}_P55prob"
  if [ "$(ls "$OUT"/*.npz 2>/dev/null | wc -l)" -lt 40 ]; then
    nnUNet_results="$RES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_${sp}_722" -o "$OUT" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
      --disable_tta --save_probabilities -npp 2 -nps 2 \
      > "$E/f2_${sp}_pred.log" 2>&1 || { echo "[F2] $sp 추론 실패" >> "$ST"; exit 1; }
  fi
done
$PY - <<'PYEOF' >> "$ST" 2>&1
import os, sys, glob, numpy as np, nibabel as nib
sys.path.insert(0, os.environ["TOPANEU_ROOT"] + "/code/sblee/nnunet/scripts")
from scipy import ndimage as ndi
import d9xx_lib as L
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
GT = L.TOPANEU_ROOT / "dataset" / "TopAneu" / "location_masks"
_, val_ids, test_ids = L.case_ids_by_split()
print("\n### F2 검출확신 진단 — P55ff blob 별 (동맥류 채널 확률)")
for sp, ids in (("test", test_ids), ("val", val_ids)):
    hall, real = [], []
    for cid in ids:
        fb = P / f"aneu_{sp}_P55ff" / f"{cid}.nii.gz"
        fp_ = P / f"aneu_{sp}_P55prob" / f"{cid}.npz"
        if not fb.exists() or not fp_.exists(): continue
        a = np.asarray(nib.load(str(fb)).dataobj) > 0
        if not a.any(): continue
        pr = np.load(fp_)["probabilities"]          # (C,z,y,x) — 라벨2=동맥류
        pa = pr[2] if pr.shape[0] > 2 else pr[-1]
        pa = np.transpose(pa, (2, 1, 0)) if pa.shape != a.shape else pa
        if pa.shape != a.shape: continue
        g = np.asarray(nib.load(str(GT / f"{cid}.nii.gz")).dataobj) > 0
        lab, k = ndi.label(a)
        for j in range(1, k + 1):
            m = lab == j
            (real if (g & m).any() else hall).append(float(pa[m].mean()))
    hall, real = np.array(hall), np.array(real)
    if len(hall) and len(real):
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.ones(len(real)), np.zeros(len(hall))]
        s = np.r_[real, hall]
        print(f"  [{sp}] 실병변 {len(real)} (검출확신 중앙 {np.median(real):.3f})"
              f" · 환각 {len(hall)} (중앙 {np.median(hall):.3f}) · 분리 AUC {roc_auc_score(y, s):.3f}")
    else:
        print(f"  [{sp}] 표본 부족 real {len(real)} hall {len(hall)}")
PYEOF
echo "F2_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$ST"
