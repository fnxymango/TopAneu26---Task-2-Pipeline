#!/usr/bin/env bash
# B2 — 번들 레시피(stock nnUNetPlans · PlainConvUNet)로 개정판 데이터 5폴드 재학습.
#      P5(ResEncL)는 08-29 e2e 에서 기각된 레시피였으므로 프로덕션 후보에서 제외한다.
#      B1(번들 모델 + 새 GT 재채점) 이 끝난 뒤에 시작한다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; B2=$E/B2_bundle_recipe_retrain
ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
PP=$R/nnunet/nnUNet_preprocessed/Dataset722_TopAneuPjh3cls417
P1PLANS=$E/P1_pjh3cls_stock250_iso04_f0/results/Dataset722_TopAneuPjh3cls417/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/plans.json
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
export PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet:${PYTHONPATH:-}
exec 9>"$D/b2.lock"; flock -n 9 || { echo "[b2] 이미 실행중" >> "$D/b2_dup.log"; exit 0; }
echo $$ > "$D/B2.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b2] $*" | tee -a "$ST"; }
cd "$S" || exit 1; mkdir -p "$B2"/{raw_results,logs}

log "B1 완료 대기"
for i in $(seq 1 720); do [ -f "$D/.done_b1" ] && break; sleep 60; done
[ -f "$D/.done_b1" ] || { log "★B1 미완 — B2 취소"; exit 1; }

# ---------------- 1) plans 복원 + 전처리 (약 2.5시간)
if [ ! -f "$D/.done_b2pp" ]; then
  cp "$P1PLANS" "$PP/nnUNetPlans.json" || { log "★plans 복원 실패"; exit 1; }
  log "plans 복원 (P1/번들 본, PlainConvUNet · 0.4iso · 112x160x128)"
  DID=$("$PY" -c "import json;print(json.load(open('$PP/nnUNetPlans.json'))['configurations']['3d_fullres']['data_identifier'])")
  n=$(ls "$PP/$DID"/*.b2nd 2>/dev/null | wc -l)
  if [ "$n" -lt 415 ]; then
    log "전처리 시작 (data_identifier=$DID · 약 2.5시간)"
    "$ENVBIN/nnUNetv2_preprocess" -d 722 -c 3d_fullres -plans_name nnUNetPlans -np 6 \
      > "$B2/logs/preprocess.log" 2>&1 || { log "★전처리 실패"; exit 1; }
  fi
  n=$(ls "$PP/$DID"/*.b2nd 2>/dev/null | wc -l)
  [ "$n" -ge 415 ] || { log "★전처리 $n/415 부족"; exit 1; }
  log "전처리 완료 · $n 건 · 여유 $(df -h / | tail -1 | awk '{print $4}')"
  "$PY" -c "
import json;s=json.load(open('$PP/splits_final.json'))
print('  splits '+str(len(s))+'폴드  '+' '.join('f%d:tr%d/va%d'%(i,len(f['train']),len(f['val'])) for i,f in enumerate(s)))" | tee -a "$ST"
  touch "$D/.done_b2pp"
fi

# ---------------- 2) 5폴드 학습 (GPU0: 0,1,2 / GPU1: 3,4)
EXP=B2_bundle_recipe_retrain
CK=$E/$EXP/results/Dataset722_TopAneuPjh3cls417/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
train(){ local g=$1; shift
  for f in $@; do
    if [ -f "$CK/fold_$f/checkpoint_final.pth" ]; then log "fold$f 이미 완료"; continue; fi
    local CONT=""; [ -f "$CK/fold_$f/checkpoint_latest.pth" ] && { CONT="--c"; log "fold$f 이어받음"; }
    log "fold$f 학습 시작 (GPU$g)"
    GPU=$g NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" \
      bash "$S/run_experiment.sh" 722 3d_fullres $f "$EXP" -p nnUNetPlans -tr nnUNetTrainer_250epochs $CONT \
      >> "$B2/logs/train_f$f.log" 2>&1
    if [ ! -f "$CK/fold_$f/checkpoint_final.pth" ] && [ -n "$CONT" ]; then
      # --c 재개가 실패했다 = checkpoint_latest 가 손상됐을 가능성. 치우고 처음부터 1회 재시도
      log "fold$f 재개 실패 — checkpoint_latest 격리 후 처음부터 재시도"
      mv "$CK/fold_$f/checkpoint_latest.pth" "$CK/fold_$f/checkpoint_latest.CORRUPT.$(date +%s)" 2>/dev/null
      GPU=$g NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" \
        bash "$S/run_experiment.sh" 722 3d_fullres $f "$EXP" -p nnUNetPlans -tr nnUNetTrainer_250epochs \
        >> "$B2/logs/train_f$f.log" 2>&1
    fi
    [ -f "$CK/fold_$f/checkpoint_final.pth" ] || { log "★fold$f 실패"; exit 1; }
    log "fold$f 완료"
  done
}
( train 0 0 1 2 ) & G0=$!
( train 1 3 4 )   & G1=$!
wait $G0 || exit 1
wait $G1 || exit 1
log "5폴드 학습 완료"; touch "$D/.done_b2train"

# ---------------- 3) e2e — 번들 파라미터 그대로, 검출기만 새 가중치
if [ ! -f "$D/.done_b2pred" ]; then
  [ -f "$A/e11_feat_hyb_ov_NEW.json" ] || { log "★신 GT 학습피처 없음 — C1 산출물 필요"; exit 1; }
  log "e2e 추론 (신 검출기 5폴드)"
  ( nnUNet_results="$E/$EXP/results" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_test_722" -o "$P/aneu_test_b2" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
      > "$B2/logs/pred_test.log" 2>&1 ) & T=$!
  ( nnUNet_results="$E/$EXP/results" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val_722" -o "$P/aneu_val_b2" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
      > "$B2/logs/pred_val.log" 2>&1 ) & V=$!
  wait $T || { log "★test 추론 실패"; exit 1; }
  wait $V || { log "★val 추론 실패"; exit 1; }
  touch "$D/.done_b2pred"
fi
for sp in test val; do
  "$PY" - "$P/aneu_${sp}_b2" "$P/aneu_${sp}_b2_bin" <<'PYEOF' >> "$B2/logs/c7.log" 2>&1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  "$PY" -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_b2_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag b2_${sp} --save-best "$P/aneu_${sp}_b2ff" --force-cfg "5,1.0" \
      >> "$B2/logs/c7.log" 2>&1 || { log "★c7 $sp 실패"; exit 1; }
done
log "c7 완료"
# 분류기 2종:
#   b2     구 GT 학습피처  — 검출기 레버만 분리해 보기 위한 대조군
#   b2new  신 GT 학습피처  — 검출기·분류기 둘 다 개정판으로 재학습한 프로덕션 후보
clf(){ local sp=$1 sd=$2 feat=$3 tag=$4 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  "$PY" -u c5_location_v2.py eval --train-feat "$feat" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b2ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tag}_${sp}_s${sd}" > "$B2/logs/clf_${tag}_${sp}_s${sd}.log" 2>&1
}
for sp in test val; do for sd in 0 1 2 3 4; do
  clf $sp $sd "$A/e11_feat_hyb_ov.json"     b2    &
  clf $sp $sd "$A/e11_feat_hyb_ov_NEW.json" b2new &
done; done
wait
cp "$A"/c5_eval_*_b2_*_s*.json "$A"/c5_percase_*_b2_*_s*.json "$B2/raw_results/" 2>/dev/null
cp "$A"/c5_eval_*_b2new_*_s*.json "$A"/c5_percase_*_b2new_*_s*.json "$B2/raw_results/" 2>/dev/null
log "B2 완료 · 분류기 20런 (b2=구피처 · b2new=신피처)"
touch "$D/.done_b2"
