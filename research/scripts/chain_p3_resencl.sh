#!/usr/bin/env bash
# P3 — pjh 레시피(Dataset722 · 0.4mm 등방 · noNorm · 3-class) 에 ResEncL 아키텍처만 얹는다 (2026-08-28).
#
# 바꾸는 레버: PlainConvUNet → ResidualEncoderUNet  (딱 하나)
# 고정: spacing 0.4mm 등방 · NoNormalization · patch 112x160x128 · batch 2 · 250ep
#       data_identifier 를 stock 과 동일하게 둬서 **기존 전처리본을 재사용**한다(재전처리 0).
#
# 왜 ResEncL 플래너 spacing(0.5x0.382x0.382) 을 안 쓰나:
#   그건 A6-2 가 쓰던 바로 그 spacing 이고, A6-2(ResEncL+adaptive norm)는 pjh stock 에 졌다.
#   (A6-2 민감도 0.814/FP 1.26  vs  X5 0.860/FP 0.43)
#   이긴 쪽의 핵심(3-class·robust z·0.4mm)을 버리면 안 된다. ABL(P2)에서 이득이 손실 축이
#   아니라 이 프레이밍에서 온다는 것을 이미 분리해 뒀다.
#
# ★ 게이트 (결과 보기 전 고정): val 42 fold0 에서 민감도 >= 0.860 AND FP/case <= 0.86
#   (FP 2배 이내면 교체가 이긴다 — A8 에서 확립)
#   통과 시에만 5폴드 확대 → test·val 두 집합 e2e 판정.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; E=$R/experiments; P=$E/_c1_realpred
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
EXP=P3_pjh3cls_resencl_iso04_f0; PLANS=nnUNetResEncUNetLPlans722iso04; TR=nnUNetTrainer_250epochs
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
export nnUNet_results=$E/$EXP/results TOPANEU_ROOT=$R
mkdir -p "$nnUNet_results"; cd $S
ST=$E/p3_status.log
log(){ echo "[P3 $(TZ=Asia/Seoul date +'%m-%d %H:%M')] $*" | tee -a "$ST"; }

log "학습 시작 — fold0 · ResEncL 101.9M · 250ep (GPU0)"
CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_train" 722 3d_fullres 0 -p "$PLANS" -tr "$TR" \
  > "$E/p3_train_f0.log" 2>&1
CK="$nnUNet_results/Dataset722_TopAneuPjh3cls417/${TR}__${PLANS}__3d_fullres/fold_0/checkpoint_best.pth"
[ -f "$CK" ] || { log "★학습 실패 — 체크포인트 없음"; exit 1; }
log "학습 완료 · $(du -h "$CK"|cut -f1)"

log "val 42 추론 (in_val_722 = robust z 입력, noNorm 함정 주의)"
OUT=$P/aneu_val_p3; BIN=$P/aneu_val_p3_bin
CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" -i "$P/in_val_722" -o "$OUT" \
  -d 722 -c 3d_fullres -f 0 -p "$PLANS" -tr "$TR" -chk checkpoint_best.pth \
  --disable_tta -npp 2 -nps 2 > "$E/p3_val_pred.log" 2>&1 || { log "★추론 실패"; exit 1; }
$PY - "$OUT" "$BIN" <<'PYEOF' || exit 1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF

log "c7 스윕 (게이트 판정용)"
$PY -u c7_detect_postproc.py --aneu-dir "$BIN" --vessel-dir "$P/vespp_val" \
    --split val --tag p3_val > "$E/p3_val_c7.txt" 2>&1 || { log "★c7 실패"; exit 1; }

$PY - <<'PYEOF' | tee -a "$ST"
import re
best=None
for L in open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/p3_val_c7.txt",errors="replace"):
    m=re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\((\d+)/(\d+)\)\s+([\d.]+)\s+(\d+)",L)
    if not m: continue
    s=float(m.group(3)); fp=float(m.group(6))
    if best is None or (s,-fp)>(best[0],-best[1]): best=(s,fp,int(m.group(1)),float(m.group(2)))
print("\n=== P3 게이트 판정 (val 42 fold0) ===")
print(f"  P3 ResEncL : 민감도 {best[0]:.3f} · FP/case {best[1]:.2f} · 최적 cfg min_vox={best[2]} dist={best[3]}")
print(f"  X5 기준선  : 민감도 0.837 (36/43) · FP/case 0.17 — cfg 5/1.0 실측. 이전 0.860/0.43 은 오기, X5_BASELINE.md 참조")
ok = best[0] >= 0.860 and best[1] <= 0.86
print(f"  기준: 민감도 >= 0.860 AND FP/case <= 0.86")
print(f"  판정: {'PASS → 5폴드 확대' if ok else 'FAIL → 중단'}")
open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/p3_gate.verdict","w").write("PASS" if ok else "FAIL")
PYEOF
log "P3_GATE_DONE"
