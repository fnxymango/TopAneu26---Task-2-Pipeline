#!/usr/bin/env bash
# P3 val 중간 판정 — val 추론이 test 보다 먼저 끝나므로 그 시점에 val 만 먼저 읽는다.
# 본 체인과 충돌하지 않도록 출력 디렉터리·태그를 전부 _mid 로 분리한다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
cd $S; ST=$E/p3_status.log
log(){ echo "[P3-mid $(TZ=Asia/Seoul date +'%m-%d %H:%M')] $*" | tee -a "$ST"; }

# val 추론 프로세스가 사라질 때까지 대기 (파일 개수가 아니라 프로세스 종료를 본다 —
# 마지막 파일이 쓰이는 중일 수 있다). 자기/부모 PID 는 제외하고 argv 를 정확히 본다.
while :; do
  alive=0
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    if tr '\0' '\n' < "$p/cmdline" 2>/dev/null | grep -qx -- "$P/aneu_val_p3x5"; then alive=1; break; fi
  done
  [ "$alive" = 0 ] && break
  sleep 30
done
n=$(ls "$P/aneu_val_p3x5"/*.nii.gz 2>/dev/null | wc -l)
[ "$n" -eq 42 ] || { log "★val 중간판정 중단 — 추론 $n/42"; exit 1; }
log "val 추론 42/42 확인 — 중간 판정 시작"

$PY - "$P/aneu_val_p3x5" "$P/aneu_val_p3x5_midbin" <<'PYEOF'
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
$PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_val_p3x5_midbin" --vessel-dir "$P/vespp_val" \
    --split val --tag p3x5mid_val --save-best "$P/aneu_val_p3x5midff" --force-cfg "5,1.0" \
    > "$E/p3mid_val_c7.txt" 2>&1 || { log "★중간 c7 실패"; exit 1; }
for sd in 0 1 2 3 4; do
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split val \
    --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --aneurysm-pred-dir "$P/aneu_val_p3x5midff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "p3x5mid_val_s${sd}" > "$E/p3x5mid_val_s${sd}.log" 2>&1 &
done
wait
$PY - <<'PYEOF' | tee -a "$ST"
import json,glob,re,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
E="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/"
b=None
for L in open(E+"p3mid_val_c7.txt",errors="replace"):
    m=re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\((\d+)/(\d+)\)\s+([\d.]+)\s+(\d+)",L)
    if m and m.group(1)=="5" and m.group(2)=="1.0": b=(float(m.group(3)),m.group(4),m.group(5),float(m.group(6)))
print("\n=== P3 val 중간 판정 (5폴드, cfg 5/1.0) ===")
if b: print(f"  검출: 민감도 {b[0]:.3f} ({b[1]}/{b[2]}) · FP/case {b[3]:.2f}   [X5 5폴드: 0.860 (37/43) · 0.43]")
fs=sorted(glob.glob(f"{A}/c5_eval_val_p3x5mid_val_s*.json"))
if not fs: print("  e2e 결과 없음"); raise SystemExit
sds=[int(f.split("_s")[-1][:-5]) for f in fs]
for key,lab in (("adjusted_div_present","covered_gt"),("official_div52","official/52")):
    x=np.array([json.load(open(f))[key]["MCC"] for f in fs])
    g=np.array([json.load(open(f"{A}/c5_eval_val_x5g_val_s{s}.json"))[key]["MCC"] for s in sds])
    d=x-g
    print(f"  {lab:11s} P3 {x.mean():.4f}±{x.std(ddof=1):.4f}  X5 {g.mean():.4f}  Δ{d.mean():+.4f}  양수 {int((d>0).sum())}/{len(d)}")
print("  ※ val 한 집합뿐 — 채택은 test 83 까지 본 뒤에만 판단한다")
PYEOF
log "P3_MID_DONE"
