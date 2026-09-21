#!/usr/bin/env bash
# ABL — 손실 축 분리 대조군 (2026-08-25). GPU1.
#
# pjh 재현(P5)은 다섯 축을 한꺼번에 바꾼다: 3-class · 스톡손실 · robust z · 0.4mm iso · PlainConvUNet.
# 이겨도 져도 **뭐 때문인지 모른다.** 그래서 같은 데이터(Dataset722)·같은 plans(nnUNetPlans,
# 따라서 같은 아키텍처 PlainConvUNet)·같은 epoch(250) 위에서 **손실만** 우리 것으로 바꾼다.
#
#   P5   nnUNetTrainer_250epochs   (Dice+CE, oversample 0.33)   ← pjh 레시피
#   ABL  nnUNetTrainerTverskyTopkCE (Tversky+TopK, oversample 0.60) ← 우리 손실
#        나머지 전부 동일  ->  차이 = 손실(+오버샘플)
#
# 왜 이게 필요한가: A7(Tversky FN벌점 2배)과 D1(TTA)이 둘 다 민감도를 못 올렸다.
# "우리 커스텀 손실이 recall 을 깎아먹는가" 가 남은 가설인데, 이 대조군이 그걸 직접 답한다.
# 새 전처리 없음 — Dataset722 를 P5 와 공유한다(디스크 추가 0).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
P="$E/_c1_realpred"; PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
DS=722; DSN="Dataset722_TopAneuPjh3cls417"
EXP="P2_pjh3cls_ourloss_iso04_f0"; TR=nnUNetTrainerTverskyTopkCE
STATUS="$E/abl_status.log"
export nnUNet_raw="$R/nnunet/nnUNet_raw" nnUNet_preprocessed="$R/nnunet/nnUNet_preprocessed"
export TOPANEU_ROOT="$R"
cd "$S" || exit 1
st(){ echo "[STEP] $*" | tee -a "$STATUS"; }
er(){ echo "[FAIL] $*" | tee -a "$STATUS"; exit 1; }

RES="$E/$EXP/results"
CK="$RES/$DSN/${TR}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
st "ABL 시작 · 손실만 우리 것 (Tversky+TopK) · Dataset722 · GPU1 · 250epoch"
if [ ! -f "$CK" ]; then
  GPU=1 NPROC=4 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" \
    bash "$S/run_experiment.sh" $DS 3d_fullres 0 "$EXP" -tr "$TR" \
    > "$E/abl_train.log" 2>&1
fi
[ -f "$CK" ] || er "학습 미완 — $E/abl_train.log"
st "ABL 학습 완료"

st "ABL 추론 · val 42 → 라벨2 추출 → c7"
OUT="$P/aneu_val_ablourloss"
if [ "$(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)" -lt 42 ]; then
  nnUNet_results="$RES" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
    -i "$P/in_val" -o "$OUT" -d $DS -c 3d_fullres -f 0 -tr "$TR" \
    -chk checkpoint_best.pth --continue_prediction -npp 2 -nps 2 \
    > "$E/abl_pred.log" 2>&1 || er "추론 실패"
fi
BIN="$P/aneu_val_ablourloss_bin"
$PY - "$OUT" "$BIN" <<'PYEOF' || er "라벨2 추출 실패"
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True)
n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
$PY -u c7_detect_postproc.py --aneu-dir "$BIN" --vessel-dir "$P/vespp_val" \
    --split val --tag abl_ourloss_val > "$E/abl_c7.txt" 2>&1 || er "c7 실패"
st "ABL 완료"
echo "=== 손실 축 대조 (val 42) ===" | tee -a "$STATUS"
echo "--- ABL: 우리 손실(Tversky+TopK, oversample .60) ---" | tee -a "$STATUS"
sed -n '/min_vox/,$p' "$E/abl_c7.txt" | head -8 | tee -a "$STATUS"
echo "--- P5: pjh 스톡손실(Dice+CE, oversample .33) ---" | tee -a "$STATUS"
sed -n '/min_vox/,$p' "$E/pjh_p6_c7.txt" 2>/dev/null | head -8 | tee -a "$STATUS"
echo "--- A6-2 fold0 (우리 원래 데이터+손실) ---" | tee -a "$STATUS"
sed -n '/min_vox/,$p' "$E/q_g1b_aneu_val_a62.txt" 2>/dev/null | head -8 | tee -a "$STATUS"
st "ABL 끝"
