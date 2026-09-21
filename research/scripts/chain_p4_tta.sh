#!/usr/bin/env bash
# P4 — 기존 X5 검출기에 TTA(미러링 8배) 를 켜본다 (2026-08-28). 학습 없음, 추론만 바뀜.
# 배경: 지금까지 29회 추론 전부 --disable_tta 였다(속도). 체크포인트엔 mirroring axes (0,1,2) 가
#       설정돼 있어 TTA 가 실제로 작동한다. 그 손해를 한 번도 측정한 적이 없다.
# ★ 게이트(사전 고정): val 42 에서 민감도 >= 0.860 AND FP/case <= 0.86  → 통과시에만 test e2e
#   최종 채택은 늘 그렇듯 test·val 두 집합 e2e 동시 + .
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
PLANS=nnUNetPlans; TR=nnUNetTrainer_250epochs
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed TOPANEU_ROOT=$R
RES=$E/P1_pjh3cls_stock250_iso04_f0/results
ST=$E/p4_status.log; cd $S
log(){ echo "[P4 $(TZ=Asia/Seoul date +'%m-%d %H:%M')] $*" | tee -a "$ST"; }
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

log "val 42 · 5폴드 TTA 추론 시작 (GPU1) — --disable_tta 제거"
nnUNet_results=$RES CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
  -i "$P/in_val_722" -o "$P/aneu_val_P55tta" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
  -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth -npp 2 -nps 2 \
  > "$E/p4_val_pred.log" 2>&1 || { log "★val 추론 실패"; exit 1; }
bin2 "$P/aneu_val_P55tta" "$P/aneu_val_P55tta_bin"
log "val c7 스윕"
$PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_val_P55tta_bin" --vessel-dir "$P/vespp_val" \
    --split val --tag p4_val --save-best "$P/aneu_val_P55ttaff" --force-cfg "5,1.0" \
    > "$E/p4_val_c7.txt" 2>&1 || { log "★c7 실패"; exit 1; }

$PY - <<'PYEOF' | tee -a "$ST"
import re
best=None
for L in open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/p4_val_c7.txt",errors="replace"):
    m=re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\((\d+)/(\d+)\)\s+([\d.]+)\s+(\d+)",L)
    if not m: continue
    s=float(m.group(3)); fp=float(m.group(6))
    if best is None or (s,-fp)>(best[0],-best[1]): best=(s,fp,int(m.group(1)),float(m.group(2)))
print("\n=== P4 TTA 게이트 (val 42) ===")
print(f"  TTA 켬    : 민감도 {best[0]:.3f} · FP/case {best[1]:.2f} · 최적 min_vox={best[2]} dist={best[3]}")
print(f"  X5 기준선 : 민감도 0.837 (36/43) · FP/case 0.17 — cfg 5/1.0 실측. 이전 0.860/0.43 은 오기, X5_BASELINE.md 참조")
ok = best[0] >= 0.860 and best[1] <= 0.86
print(f"  판정: {'PASS → test e2e 진행' if ok else 'FAIL → 중단'}")
open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/p4_gate.verdict","w").write("PASS" if ok else "FAIL")
PYEOF
[ "$(cat $E/p4_gate.verdict)" = PASS ] || { log "P4_DONE (게이트 미통과)"; exit 0; }

log "test 83 · 5폴드 TTA 추론"
nnUNet_results=$RES CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
  -i "$P/in_test_722" -o "$P/aneu_test_P55tta" -d 722 -c 3d_fullres -f 0 1 2 3 4 \
  -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth -npp 2 -nps 2 \
  > "$E/p4_test_pred.log" 2>&1 || { log "★test 추론 실패"; exit 1; }
bin2 "$P/aneu_test_P55tta" "$P/aneu_test_P55tta_bin"
$PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_test_P55tta_bin" --vessel-dir "$P/vespp_test" \
    --split test --tag p4_test --save-best "$P/aneu_test_P55ttaff" --force-cfg "5,1.0" \
    > "$E/p4_test_c7.txt" 2>&1 || { log "★c7 실패"; exit 1; }

log "e2e 5시드 (gC 위에서, test·val)"
one(){ local sp=$1 sd=$2 ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ttaff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "p4tta_${sp}_s${sd}" > "$E/p4tta_${sp}_s${sd}.log" 2>&1
}
for sd in 0 1 2 3 4; do one test $sd & one val $sd & done
wait
$PY - <<'PYEOF' | tee -a "$ST"
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
X5={"test":"c5_eval_test_x5g_test_s%d.json","val":"c5_eval_val_x5g_val_s%d.json"}
ok=True; print("\n### P4 TTA + gC vs X5+gC (현행 최고)")
for sp in ("test","val"):
    fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_p4tta_{sp}_s*.json"))
    if not fs: print(f"  {sp}: 결과 없음"); ok=False; continue
    sds=[int(f.split("_s")[-1][:-5]) for f in fs]
    x=np.array([json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs])
    g=np.array([json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"]["MCC"] for s in sds])
    d=x-g; sd_=d.std(ddof=1) if len(d)>1 else 0
    t=d.mean()/(sd_/np.sqrt(len(d))) if sd_>1e-12 else float("nan")
    nl=json.load(open(fs[0]))["n_lesions_predicted"]
    print(f"  {sp:5s} {x.mean():.4f}±{x.std(ddof=1):.4f}  vs x5g {g.mean():.4f}  Δ{d.mean():+.4f} t{t:+.2f} {int((d>0).sum())}/{len(d)}  blob {nl}")
    if d.mean()<=0: ok=False
print(f"판정: {'PASS' if ok else 'FAIL'}")
PYEOF
log "P4_DONE"
