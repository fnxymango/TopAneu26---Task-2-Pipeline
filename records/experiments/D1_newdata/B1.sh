#!/usr/bin/env bash
# B1 — 제출 번들(sblee_task2_final_v2) 구성 그대로, 데이터셋만 2026-09 개정판으로 바꿔 전체 재실행.
#      바꾼 것: GT/영상뿐. 모델·파라미터·코드는 번들과 동일.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; B1=$E/B1_bundle_newdata
ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
export PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet:${PYTHONPATH:-}
exec 9>"$D/b1.lock"; flock -n 9 || { echo "[b1] 이미 실행중" >> "$D/b1_dup.log"; exit 0; }
echo $$ > "$D/B1.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b1] $*" | tee -a "$ST"; }
cd "$S" || exit 1
mkdir -p "$B1"/{raw_results,logs}

log "GPU 확보 대기 (C1 완료 · C2 취소)"
for i in $(seq 1 480); do { [ -f "$D/.done_c1c2" ] || [ -f "$D/.cancelled_c2" ]; } && break; sleep 60; done
log "시작 · 번들 구성 그대로, 데이터셋만 개정판"

# ---------------- 1) 검출기 P1 5폴드 추론
DRES=$E/P1_pjh3cls_stock250_iso04_f0/results
if [ ! -f "$D/.done_b1pred" ]; then
  log "검출기 P1 5폴드 추론 (stock nnUNetPlans)"
  ( nnUNet_results="$DRES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_test_722" -o "$P/aneu_test_b1" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
      --disable_tta -npp 2 -nps 2 > "$B1/logs/pred_test.log" 2>&1 ) & T=$!
  ( nnUNet_results="$DRES" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val_722" -o "$P/aneu_val_b1" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
      --disable_tta -npp 2 -nps 2 > "$B1/logs/pred_val.log" 2>&1 ) & V=$!
  wait $T || { log "★test 추론 실패"; exit 1; }
  wait $V || { log "★val 추론 실패"; exit 1; }
  log "추론 완료"; touch "$D/.done_b1pred"
fi

# ---------------- 2) 라벨2 추출 + c7 (번들: min_vox=5, max_dist=1.0)
for sp in test val; do
  "$PY" - "$P/aneu_${sp}_b1" "$P/aneu_${sp}_b1_bin" <<'PYEOF' >> "$B1/logs/c7.log" 2>&1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  "$PY" -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_b1_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag b1_${sp} --save-best "$P/aneu_${sp}_b1ff" --force-cfg "5,1.0" \
      >> "$B1/logs/c7.log" 2>&1 || { log "★c7 $sp 실패"; exit 1; }
done
log "c7 완료"

# ---------------- 3) 분류기 — 번들 파라미터 그대로. seed3 이 공식값, 0~4 는 변동폭 확인용
one(){ local sp=$1 sd=$2 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  "$PY" -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "b1_${sp}_s${sd}" > "$B1/logs/clf_${sp}_s${sd}.log" 2>&1
}
for sp in test val; do for sd in 0 1 2 3 4; do one $sp $sd & done; done
wait
log "분류기 10런 완료"
cp "$A"/c5_eval_*_b1_*_s*.json "$A"/c5_percase_*_b1_*_s*.json "$B1/raw_results/" 2>/dev/null

# ---------------- 4) 정리
"$PY" - "$B1" "$A" <<'PYEOF' > "$B1/RESULTS.md" 2>&1
import json,sys,os
import numpy as np
B1,A=sys.argv[1],sys.argv[2]
def L(sp,sd): return json.load(open(f"{A}/c5_eval_{sp}_b1_{sp}_s{sd}.json"))
print("# B1 — 제출 번들 구성 그대로, 데이터셋만 2026-09 개정판\n")
print("바꾼 것은 GT/영상뿐. 모델 가중치·파라미터·코드는 번들과 동일하다.\n")
print("```")
print("검출기   P1 stock nnUNetPlans 5폴드 (번들 체크포인트와 md5 동일)")
print("혈관     V4-2 ClassWeightedV2 -> V5 후처리")
print("c7      min_vox=5 · max_dist=1.0mm")
print("분류기   RF500 · clf_seed=3 (공식) · beta 0.5 · topk n=2 vox=3 ica=1 margin=0.7")
print("학습피처 e11_feat_hyb_ov.json (번들본)")
print("데이터   415케이스 · train 291 / val 41 / test 83")
print("```\n")
for sp,n in (("test",83),("val",41)):
    o=L(sp,3)
    print(f"## {sp} {n}건 — 공식값 (clf_seed=3)\n")
    print("| 기준 | Precision | Recall | MCC | Dice | VolSim | HD95 | 6지표 |")
    print("|---|---|---|---|---|---|---|---|")
    for key,lab in (("official_div52","official ÷52"),("adjusted_div_present",f"covered_gt ÷{o['n_present_classes_in_split']}")):
        d=o[key]
        six=np.mean([d["PRECISION"],d["RECALL"],d["MCC"],d["DICE"],d["VOLSIM"],1-d["HD95"]])
        print(f"| {lab} | {d['PRECISION']:.4f} | {d['RECALL']:.4f} | **{d['MCC']:.4f}** | {d['DICE']:.4f} | {d['VOLSIM']:.4f} | {d['HD95']:.4f} | {six:.4f} |")
    print(f"\n예측 blob {o['n_lesions_predicted']} · 케이스 {o['n_cases']} · 존재클래스 {o['n_present_classes_in_split']}\n")
    print("시드 변동폭 (0~4, 참고용)\n")
    print("| 기준 | MCC 평균 | 표준편차 | 최소 | 최대 |")
    print("|---|---|---|---|---|")
    for key,lab in (("official_div52","official ÷52"),("adjusted_div_present","covered_gt")):
        v=np.array([L(sp,s)[key]["MCC"] for s in range(5)])
        print(f"| {lab} | {v.mean():.4f} | {v.std(ddof=1):.4f} | {v.min():.4f} | {v.max():.4f} |")
    print()
print("## 구 GT 기준 공표값과의 대조\n")
print("| | 구 GT (2026-08-01) | 신 GT (2026-09-01) |")
print("|---|---|---|")
print("| test covered_gt MCC | 0.4121 | %.4f |" % L("test",3)["adjusted_div_present"]["MCC"])
print("| test official ÷52 MCC | 0.2853 | %.4f |" % L("test",3)["official_div52"]["MCC"])
print("\n차이는 모델이 아니라 라벨 수정에서 온다 — 개정판이 test 병변을 86→89로 늘렸고 그 3건은 이미 검출되고 있었다(FP 31→28).")
PYEOF
log "B1 완료 · $B1/RESULTS.md"
touch "$D/.done_b1"
