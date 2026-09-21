#!/usr/bin/env bash
# ENS — 계열 판정(.line_choice)을 읽어 이긴 계열 안에서 10폴드 앙상블을 만든다.
#   stock  : P1(구데이터) fold0~4 + B2(신데이터) fold0~4
#   resencl: P3(구데이터) fold0~4 + P5(신데이터) fold0~4
# 확률 npz 를 저장하지 않고, 10개 폴드를 한 results 디렉터리에 심볼릭으로 모아
# nnUNet 이 내부에서 평균하게 한다(디스크 비용 0).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; ST=$D/STATUS.log
OUT=$E/E9_ensemble10
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
export PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet:${PYTHONPATH:-}
exec 9>"$D/ens.lock"; flock -n 9 || exit 0
echo $$ > "$D/ENS.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][ens] $*" | tee -a "$ST"; }
cd "$S" || exit 1; mkdir -p "$OUT"/{raw_results,logs}

log "계열 판정 대기"
for i in $(seq 1 2880); do [ -f "$D/.done_b2eval" ] && break; sleep 60; done
[ -f "$D/.line_choice" ] || { log "★계열 판정 없음 — 취소"; exit 1; }
LINE=$(tr -d '[:space:]' < "$D/.line_choice")
log "계열 = $LINE"

SUB=Dataset722_TopAneuPjh3cls417
if [ "$LINE" = "stock" ]; then
  PLANS=nnUNetPlans; TR=nnUNetTrainer_250epochs
  OLDR=$E/P1_pjh3cls_stock250_iso04_f0/results/$SUB/${TR}__${PLANS}__3d_fullres
  NEWR=$E/B2_bundle_recipe_retrain/results/$SUB/${TR}__${PLANS}__3d_fullres
else
  PLANS=nnUNetResEncUNetLPlans722iso04; TR=nnUNetTrainer_250epochs
  OLDR=$E/P3_pjh3cls_resencl_iso04_f0/results/$SUB/${TR}__${PLANS}__3d_fullres
  NEWR=$E/P5_newdata_detector/results/$SUB/${TR}__${PLANS}__3d_fullres
fi
ENS=$OUT/results/$SUB/${TR}__${PLANS}__3d_fullres
mkdir -p "$ENS"
for f in 0 1 2 3 4; do ln -sfn "$OLDR/fold_$f" "$ENS/fold_$f"; ln -sfn "$NEWR/fold_$f" "$ENS/fold_$((f+5))"; done
cp "$OLDR/plans.json" "$OLDR/dataset.json" "$ENS/" 2>/dev/null
miss=0; for f in 0 1 2 3 4 5 6 7 8 9; do [ -f "$ENS/fold_$f/checkpoint_best.pth" ] || { log "★fold_$f 체크포인트 없음"; miss=1; }; done
[ "$miss" = 0 ] || exit 1
log "10폴드 구성 완료 ($LINE · $PLANS)"

if [ ! -f "$D/.done_enspred" ]; then
  log "10폴드 추론"
  ( nnUNet_results="$OUT/results" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_test_722" -o "$P/aneu_test_e9" -d 722 -c 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 \
      -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
      > "$OUT/logs/pred_test.log" 2>&1 ) & T=$!
  ( nnUNet_results="$OUT/results" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val_722" -o "$P/aneu_val_e9" -d 722 -c 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 \
      -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
      > "$OUT/logs/pred_val.log" 2>&1 ) & V=$!
  wait $T || { log "★test 추론 실패"; exit 1; }
  wait $V || { log "★val 추론 실패"; exit 1; }
  touch "$D/.done_enspred"; log "추론 완료"
fi

for sp in test val; do
  "$PY" - "$P/aneu_${sp}_e9" "$P/aneu_${sp}_e9_bin" <<'PYEOF' >> "$OUT/logs/c7.log" 2>&1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  "$PY" -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_e9_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag e9_${sp} --save-best "$P/aneu_${sp}_e9ff" --force-cfg "5,1.0" \
      >> "$OUT/logs/c7.log" 2>&1 || { log "★c7 $sp 실패"; exit 1; }
done
log "c7 완료"

clf(){ local sp=$1 sd=$2 feat=$3 tag=$4 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  "$PY" -u c5_location_v2.py eval --train-feat "$feat" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_e9ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tag}_${sp}_s${sd}" > "$OUT/logs/clf_${tag}_${sp}_s${sd}.log" 2>&1
}
for sp in test val; do for sd in 0 1 2 3 4; do
  clf $sp $sd "$A/e11_feat_hyb_ov.json"     e9    &
  clf $sp $sd "$A/e11_feat_hyb_ov_NEW.json" e9new &
done; done
wait
cp "$A"/c5_eval_*_e9*_s*.json "$A"/c5_percase_*_e9*_s*.json "$OUT/raw_results/" 2>/dev/null
log "분류기 20런 완료"

{
  "$PY" "$D/record_metrics.py" e9new "E9 — $LINE 계열 10폴드 앙상블 (분류기: 개정판 피처)"
  echo "---"; echo
  "$PY" "$D/record_metrics.py" e9 "E9-대조 — 같은 앙상블, 분류기는 구 GT 피처"
  echo "---"; echo
  echo "# 단독 모델 대비 (채택 판정용)"; echo
  if [ "$LINE" = "stock" ]; then
    "$PY" "$D/cmp_tags.py" b2new e9new "B2 단독(신)" "10폴드 앙상블" | tail -n +2
    echo; "$PY" "$D/cmp_tags.py" b1 e9 "B1 단독(구)" "10폴드 앙상블(구피처)" | tail -n +2
  else
    "$PY" "$D/cmp_tags.py" cmpnew e9 "P5 단독(신)" "10폴드 앙상블" | tail -n +2
    echo; "$PY" "$D/cmp_tags.py" cmpold e9 "P3 단독(구)" "10폴드 앙상블" | tail -n +2
  fi
} > "$OUT/RESULTS.md" 2>&1
log "ENS 완료 · $OUT/RESULTS.md"
touch "$D/.done_ens"
