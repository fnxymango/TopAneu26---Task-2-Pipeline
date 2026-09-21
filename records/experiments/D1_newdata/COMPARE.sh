#!/usr/bin/env bash
# 구 검출기(P3, 구 GT 학습) vs 신 검출기(P5, 신 GT 학습) — 같은 자로 대조.
#   동결: 분류기 학습피처 e11_feat_hyb_ov · 예측혈관 vespp_* · BP그래프 · c7 force-cfg 5,1.0 · 시드 0~4
#   레버: 검출기 체크포인트 하나뿐
#   채점: 둘 다 **새 GT** (test 83 · val 41)
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; ST=$D/STATUS.log
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
PLANS=nnUNetResEncUNetLPlans722iso04; TR=nnUNetTrainer_250epochs
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
export PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet:${PYTHONPATH:-}
OLDRES=$E/P3_pjh3cls_resencl_iso04_f0/results
NEWRES=$E/P5_newdata_detector/results
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][cmp] $*" | tee -a "$ST"; }
exec 9>"$D/compare.lock"; flock -n 9 || { echo "[cmp] 이미 실행중 — 중복 취소" >> "$D/compare_dup.log"; exit 0; }
echo $$ > "$D/COMPARE.pid"
cd "$S" || exit 1

log "학습 완료 대기"
while [ ! -f "$D/.done_det_gpu0" ] || [ ! -f "$D/.done_det_gpu1" ]; do sleep 120; done
log "학습 완료 확인 — 대조 시작"

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

# ---- 추론: 구=GPU0, 신=GPU1 (각각 test 뒤 val) ----
pred(){  # $1=arm $2=results $3=gpu
  local arm=$1 res=$2 g=$3
  for sp in test val; do
    local out="$P/aneu_${sp}_cmp${arm}"
    if [ "$(ls "$out"/*.nii.gz 2>/dev/null | wc -l)" -ge 1 ]; then log "$arm/$sp 추론 이미 있음"; continue; fi
    log "$arm/$sp 5폴드 추론 (GPU$g)"
    nnUNet_results="$res" CUDA_VISIBLE_DEVICES=$g "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_${sp}_722" -o "$out" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
      -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
      > "$D/cmp_${arm}_${sp}_pred.log" 2>&1 || { log "★$arm/$sp 추론 실패"; touch "$D/.failed_compare"; return 1; }
  done
}
( pred old "$OLDRES" 0 ) & O=$!
( pred new "$NEWRES" 1 ) & N=$!
wait $O || exit 1
wait $N || exit 1
log "추론 완료"

# ---- c7 후처리 (프로덕션 설정 동결) ----
for arm in old new; do for sp in test val; do
  bin2 "$P/aneu_${sp}_cmp${arm}" "$P/aneu_${sp}_cmp${arm}_bin" >> "$D/cmp_c7.log" 2>&1
  nz=$($PY -c "
import glob,numpy as np,nibabel as nib
print(sum(1 for f in glob.glob('$P/aneu_${sp}_cmp${arm}_bin/*.nii.gz') if np.asanyarray(nib.load(f).dataobj).any()))")
  log "$arm/$sp 이진마스크 비어있지않은 케이스 $nz"
  [ "$nz" -eq 0 ] && { log "★$arm/$sp 예측에 동맥류(라벨2) 가 하나도 없다 — 중단"; touch "$D/.failed_compare"; exit 1; }
  $PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_cmp${arm}_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag cmp${arm}_${sp} --save-best "$P/aneu_${sp}_cmp${arm}ff" --force-cfg "5,1.0" \
      >> "$D/cmp_c7.log" 2>&1 || { log "★c7 $arm/$sp 실패 — 재시동 중단"; touch "$D/.failed_compare"; exit 1; }
done; done
log "c7 완료"

# ---- 분류기 5시드 × 2스플릿 × 2팔 ----
one(){ local arm=$1 sp=$2 sd=$3 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_cmp${arm}ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "cmp${arm}_${sp}_s${sd}" > "$D/cmp_${arm}_${sp}_s${sd}.log" 2>&1
}
for arm in old new; do for sp in test val; do for sd in 0 1 2 3 4; do one $arm $sp $sd & done; done; done
wait
log "분류기 20런 완료"

# ---- 판정 ----
$PY - <<'PYEOF' 2>&1 | tee "$D/COMPARE_RESULT.txt" | tee -a "$ST"
import json,glob,os
import numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
def load(arm,sp):
    out=[]
    for sd in range(5):
        p=f"{A}/c5_eval_{sp}_cmp{arm}_{sp}_s{sd}.json"
        if os.path.exists(p): out.append(json.load(open(p)))
    return out
print("="*74)
print("구 검출기(구 GT 학습) vs 신 검출기(신 GT 학습) — 둘 다 새 GT 로 채점")
print("동결: 분류기피처·예측혈관·BP그래프·c7(5,1.0)·시드0~4   레버: 검출기뿐")
verdict={}
for sp,n in (("test",83),("val",41)):
    o,nw=load("old",sp),load("new",sp)
    if not o or not nw: print(f"\n[{sp}] 결과 부족 (구 {len(o)} 신 {len(nw)})"); continue
    print(f"\n[{sp} {n}건]  blob 구 {o[0]['n_lesions_predicted']} → 신 {nw[0]['n_lesions_predicted']}"
          f" · 검출커버리지 {o[0]['detect_coverage']:.3f} → {nw[0]['detect_coverage']:.3f}")
    for key,lab in (("official_div52","official/52"),("adjusted_div_present","covered_gt")):
        for met in ("MCC","PRECISION","RECALL"):
            a=np.array([x[key][met] for x in o]); b=np.array([x[key][met] for x in nw])
            d=b-a
            star=" ★" if met=="MCC" else ""
            print(f"  {lab:11s} {met:9s} 구 {a.mean():.4f}  신 {b.mean():.4f}  Δ{d.mean():+.4f}"
                  f"  시드양수 {int((d>0).sum())}/5{star}")
            if met=="MCC": verdict[(sp,lab)]=(d.mean(),int((d>0).sum()))
        six_a=np.mean([[x[key][m] for m in ("PRECISION","RECALL","MCC","DICE","VOLSIM")]+[1-x[key]["HD95"]] for x in o])
        six_b=np.mean([[x[key][m] for m in ("PRECISION","RECALL","MCC","DICE","VOLSIM")]+[1-x[key]["HD95"]] for x in nw])
        print(f"  {lab:11s} {'6지표':9s} 구 {six_a:.4f}  신 {six_b:.4f}  Δ{six_b-six_a:+.4f}")
print("\n"+"="*74)
ok=all(v[0]>0 for k,v in verdict.items() if k[1]=="covered_gt")
for k,v in sorted(verdict.items()): print(f"  {k[0]:5s} {k[1]:11s} MCC Δ{v[0]:+.4f} (시드 {v[1]}/5)")
print(f"\n판정: 신 검출기 {'채택 — test·val covered_gt MCC 둘 다 상승' if ok else '보류 — 두 집합 동시 상승 아님'}")
PYEOF
log "COMPARE_DONE — 결과 $D/COMPARE_RESULT.txt"
touch "$D/.done_compare"
