#!/usr/bin/env bash
# C1 = 분류기 학습피처를 새 GT 로 재구축 (레버: 학습피처)
# C2 = 검출기 10폴드 앙상블 (구 P3 5폴드 + 신 P5 5폴드, 확률 평균) (레버: 검출기)
# 두 레버는 독립. 기준선은 이미 계산된 cmpold (구 검출기 + 구 피처).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
ENVBIN2=$HOME/miniconda3/envs/sbaneu2/bin
PLANS=nnUNetResEncUNetLPlans722iso04; TR=nnUNetTrainer_250epochs
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
export PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet:${PYTHONPATH:-}
exec 9>"$D/c1c2.lock"; flock -n 9 || { echo "[c1c2] 이미 실행중" >> "$D/c1c2_dup.log"; exit 0; }
echo $$ > "$D/C1C2.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][c1c2] $*" | tee -a "$ST"; }
cd "$S" || exit 1

# ================================================ Stage0 · mr_325 되살리기 (영상이 바뀐 유일 케이스)
CID=topaneu_center1_mr_325
if [ ! -f "$D/.done_s0" ]; then
  log "Stage0 · $CID 혈관예측·BP 재생성 (구 예측은 크기 불일치)"
  W=$D/s0_work; mkdir -p "$W/in" "$W/ves"
  cp "$R/dataset/TopAneu/images/${CID}_0000.nii.gz" "$W/in/"
  nnUNet_results="$E/V4-2_vessel_classweighted_417_500ep/results" CUDA_VISIBLE_DEVICES=1 \
    "$ENVBIN2/nnUNetv2_predict" -i "$W/in" -o "$W/ves" -d 800 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep -p nnUNetResEncUNetMPlans \
    -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 > "$D/s0_pred.log" 2>&1 \
    || { log "★Stage0 혈관추론 실패"; exit 1; }
  "$PY" "$S/postprocess_vessel.py" apply "$W/ves" "$W/vespp" > "$D/s0_pp.log" 2>&1 \
    || { log "★Stage0 후처리 실패"; exit 1; }
  cp "$W/vespp/${CID}.nii.gz" "$P/vespp_train/${CID}.nii.gz"
  "$PY" "$S/c4_branchpoint_graph.py" --vessel-dir "$P/vespp_train" --out "$BP/vespp_train" \
    --cases "$CID" >> "$D/s0_pp.log" 2>&1 || { log "★Stage0 BP(예측) 실패"; exit 1; }
  "$PY" "$S/c4_branchpoint_graph.py" --vessel-dir "$R/dataset/TopAneu/vessel_masks" --out "$BP/all_ref" \
    --cases "$CID" >> "$D/s0_pp.log" 2>&1 || { log "★Stage0 BP(GT) 실패"; exit 1; }
  log "Stage0 완료 · $CID 갱신"
  touch "$D/.done_s0"
fi

# ================================================ C1 · 학습피처 재구축 (CPU) — 백그라운드
( 
  if [ ! -f "$D/.done_c1feat" ]; then
    log "C1 · 학습피처 재구축 시작 (train 291, 새 GT)"
    "$PY" -u c5_location_v2.py build --split train --vessel-dir "$R/dataset/TopAneu/vessel_masks" \
      --bp-dir "$BP/all_ref" --out "$A/c10_feat_train_NEW.json" > "$D/c1_build_ref.log" 2>&1 \
      || { log "★C1 GT혈관 피처 실패"; exit 1; }
    "$PY" -u c5_location_v2.py build --split train --vessel-dir "$P/vespp_train" \
      --bp-dir "$BP/vespp_train" --out "$A/c10_feat_train_predves_NEW.json" > "$D/c1_build_prd.log" 2>&1 \
      || { log "★C1 예측혈관 피처 실패"; exit 1; }
    # 하이브리드: GT혈관 피처에서 overlap 만 예측혈관 것으로 교체 (구 e11 규칙을 역산해 확인함)
    "$PY" - "$A" <<'PYEOF' >> "$D/c1_build_prd.log" 2>&1 || { log "★C1 병합 실패"; exit 1; }
import json,sys
A=sys.argv[1]
ref=json.load(open(f"{A}/c10_feat_train_NEW.json"))
prd=json.load(open(f"{A}/c10_feat_train_predves_NEW.json"))
key=lambda r:(r["case"], r.get("lesion_mask_idx"))
pm={key(r):r for r in prd}
n=0
for r in ref:
    q=pm.get(key(r))
    if q is not None and "overlap" in q: r["overlap"]=q["overlap"]; n+=1
json.dump(ref, open(f"{A}/e11_feat_hyb_ov_NEW.json","w"))
print(f"  overlap 교체 {n}/{len(ref)}")
PYEOF
    log "C1 피처 완료 · $(basename $A)/e11_feat_hyb_ov_NEW.json"
    touch "$D/.done_c1feat"
  fi
  # 평가: 구 검출기 산출물 위에서 피처만 교체
  for sp in test val; do
    ves=vespp_$sp; bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
    for sd in 0 1 2 3 4; do
      TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
      "$PY" -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov_NEW.json" --split "$sp" \
        --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_cmpoldff" \
        --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
        --tag "c1feat_${sp}_s${sd}" > "$D/c1_${sp}_s${sd}.log" 2>&1 &
    done
  done
  wait
  log "C1 완료 · 분류기 10런"
  touch "$D/.done_c1"
) &
C1PID=$!

# ================================================ C2 · 10폴드 앙상블 (GPU)
if [ ! -f "$D/.done_c2pred" ]; then
  log "C2 · 10폴드 앙상블 추론 (구5+신5, 확률 평균)"
  ENSRES=$E/E1_ens10/results
  ( nnUNet_results="$ENSRES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_test_722" -o "$P/aneu_test_ens10" -d 722 -c 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 \
      -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
      > "$D/c2_test_pred.log" 2>&1 ) & T=$!
  ( nnUNet_results="$ENSRES" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val_722" -o "$P/aneu_val_ens10" -d 722 -c 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 \
      -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
      > "$D/c2_val_pred.log" 2>&1 ) & V=$!
  wait $T || { log "★C2 test 추론 실패"; exit 1; }
  wait $V || { log "★C2 val 추론 실패"; exit 1; }
  log "C2 추론 완료"
  touch "$D/.done_c2pred"
fi
for sp in test val; do
  "$PY" - "$P/aneu_${sp}_ens10" "$P/aneu_${sp}_ens10_bin" <<'PYEOF' >> "$D/c2_c7.log" 2>&1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  "$PY" -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_ens10_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag ens10_${sp} --save-best "$P/aneu_${sp}_ens10ff" --force-cfg "5,1.0" \
      >> "$D/c2_c7.log" 2>&1 || { log "★C2 c7 $sp 실패"; exit 1; }
done
log "C2 c7 완료"
for sp in test val; do
  ves=vespp_$sp; bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  for sd in 0 1 2 3 4; do
    TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
    "$PY" -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
      --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_ens10ff" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "ens10_${sp}_s${sd}" > "$D/c2_${sp}_s${sd}.log" 2>&1 &
  done
done
wait
log "C2 완료 · 분류기 10런"
touch "$D/.done_c2"
wait $C1PID
log "C1C2_DONE"
touch "$D/.done_c1c2"
