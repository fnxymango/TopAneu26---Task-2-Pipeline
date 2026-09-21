#!/usr/bin/env bash
# REP5F — 제출 tar 체크포인트로 **5폴드** 재추론 (2026-09-18 · 사용자 지시)
#
#  왜: 3폴드(b3ff)가 5폴드(b1ff)보다 GT 를 더 찾았다(test 72 vs 70 · val 37 vs 36).
#      해석이 두 갈래인데 지금은 못 가른다:
#        (가) 폴드를 적게 평균하면 소수 의견이 덜 희석돼 동작점이 관대해진다
#        (나) 두 산출물의 **체크포인트 출처가 다르다** — b1ff 는 이제 없는
#             `P1_pjh3cls_stock250_iso04_f0/results` 에서 나왔고 b3ff 는 제출 tar 에서 꺼냈다.
#             B1.sh 주석은 "번들과 md5 동일" 이라지만 원본이 삭제돼 확인 불가다.
#      **같은 체크포인트로 5폴드를 다시 돌리면 출처가 고정되고 폴드 수만 남는다.**
#      덤으로 5폴드의 **후처리 전** 산출물이 복원돼 단계별 TP/FP 표의 빈 칸이 채워진다.
#
#  묻는 것 (결과 보기 전 고정 · 2026-09-18 15:10 KST)
#    ① b5ff(새 5폴드) 가 b1ff(옛 5폴드) 와 **같은가** — 다르면 (나)가 실재한다는 뜻이고,
#       그동안의 5폴드 기준선 수치는 제출본과 다른 가중치였을 수 있다. 이건 보고 사항이다.
#    ② b5ff 대 b3ff 로 **폴드 수 효과만** 잰다: GT 적중 · blob 수 · 작은 blob 수.
#       (가)가 맞다면 3폴드가 적중↑ · blob↑ · 작은 blob↑ 이어야 한다.
#    분류 e2e 는 여기서 돌리지 않는다 — 검출 단계 비교가 먼저다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts
P=$E/_c1_realpred; D=$E/D1_newdata; V=$E/V1_vessel_axis
SC=/tmp/scratch
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
DETRES=$SC/det3f/models/detector
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
exec 9>"$D/rep5f.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][rep5f] $*" | tee -a "$D/STATUS.log"; }

if [ ! -f "$D/.done_rep5f_det" ]; then
  log "5폴드 재추론 (제출 tar 체크포인트 · folds 0,1,2,3,4)"
  ( nnUNet_results="$DETRES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_test_722" -o "$P/aneu_test_b5" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
      --disable_tta -npp 2 -nps 2 > "$D/rep5f_pred_test.log" 2>&1 ) & T=$!
  ( nnUNet_results="$DETRES" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val_722" -o "$P/aneu_val_b5" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
      --disable_tta -npp 2 -nps 2 > "$D/rep5f_pred_val.log" 2>&1 ) & W=$!
  wait $T || { log "★test 추론 실패"; exit 1; }
  wait $W || { log "★val 추론 실패"; exit 1; }
  touch "$D/.done_rep5f_det"; log "추론 완료"
fi

cd "$S" || exit 1
for sp in test val; do
  [ -d "$P/aneu_${sp}_b5ff" ] && [ "$(ls -1 $P/aneu_${sp}_b5ff 2>/dev/null|wc -l)" -gt 0 ] && continue
  "$PY" - "$P/aneu_${sp}_b5" "$P/aneu_${sp}_b5_bin" <<'PYEOF' >> "$D/rep5f_c7.log" 2>&1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  "$PY" -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_b5_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag b5_${sp} --save-best "$P/aneu_${sp}_b5ff" --force-cfg "5,1.0" \
      >> "$D/rep5f_c7.log" 2>&1 || { log "★c7 $sp 실패"; exit 1; }
done
log "c7(5,1.0) 완료"
touch "$D/.done_rep5f"
log "REP5F 끝 — 단계별 표 재계산 가능"
