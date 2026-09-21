#!/bin/bash
# S3 혈관 경계 섭동 증강 (2026-08-27). 라벨오류 58.5% 가 같은 혈관 인접분절인데 그 경계는 혈관 모델이
# 임의로 긋는다. 예측혈관(vespp_train)의 라벨 경계를 1복셀씩 흔든 마스크 2종에서 overlap 블록을 다시 뽑아
# 학습행에 증강으로 추가(T16 하이브리드 방식 그대로: 참조행 + overlap 만 교체). 규칙: X5+gC 대비 test·val 동시 +.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
ST=$E/chain_status.md
echo "[$(TZ=Asia/Seoul date +%H:%M)] S3 시작 — 경계 섭동 마스크 생성" >> "$ST"

$PY - <<'PYEOF' || exit 1
import os, sys, numpy as np, nibabel as nib
sys.path.insert(0, os.environ["TOPANEU_ROOT"] + "/code/sblee/nnunet/scripts")
from scipy import ndimage as ndi
import d9xx_lib as L
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
train_ids, _, _ = L.case_ids_by_split()
src = P / "vespp_train"
for k in (1, 2):
    dst = P / f"vespp_train_pert{k}"; dst.mkdir(exist_ok=True)
    rng = np.random.default_rng(100 + k); n = 0
    for cid in train_ids:
        f = src / f"{cid}.nii.gz"
        if not f.exists(): continue
        im = nib.load(str(f)); v = np.asanyarray(im.dataobj).astype(np.int16)
        fg = v > 0; out = v.copy()
        labs = [int(x) for x in np.unique(v) if x > 0]; rng.shuffle(labs)
        for lab in labs:                       # 라벨 경쟁: 절반의 라벨을 1복셀 전진시켜 인접분절 경계를 민다
            if rng.random() < 0.5: continue
            m = out == lab
            grown = ndi.binary_dilation(m, iterations=1) & fg & ~m
            out[grown] = lab
        o = nib.Nifti1Image(out.astype(v.dtype), im.affine, im.header); o.set_data_dtype(v.dtype)
        nib.save(o, str(dst / f"{cid}.nii.gz")); n += 1
    print(f"  pert{k}: {n}케이스", flush=True)
PYEOF

for k in 1 2; do
  [ -f "$A/c10_feat_train_pert$k.json" ] || \
  $PY -u c5_location_v2.py build --split train --vessel-dir "$P/vespp_train_pert$k" \
      --bp-dir "$BP/vespp_train" --out "$A/c10_feat_train_pert$k.json" > "$E/s3_build_pert$k.log" 2>&1 &
done
wait
echo "[$(TZ=Asia/Seoul date +%H:%M)] S3 피처 추출 완료 — 증강행 조립" >> "$ST"

$PY - <<'PYEOF' >> "$ST" 2>&1 || exit 1
import json, os
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
ref=json.load(open(f"{A}/e11_feat_hyb_ov.json")); key=lambda r:(r["case"],r["lesion_mask_idx"])
out=list(ref); added=0
for k in (1,2):
    pm={key(r):r for r in json.load(open(f"{A}/c10_feat_train_pert{k}.json"))}
    for r in ref:
        q=pm.get(key(r))
        if q and q["overlap"]!=r["overlap"]:
            s=dict(r); s["overlap"]=q["overlap"]; out.append(s); added+=1
json.dump(out,open(f"{A}/e11_feat_hyb_ov_s3aug.json","w"),ensure_ascii=False)
print(f"  S3 증강: 원본 {len(ref)} + 섭동행 {added} = {len(out)}")
PYEOF

one(){ local sp=$1 sd=$2; local ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov_s3aug.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "s3aug_${sp}_s${sd}" > "$E/s3aug_${sp}_s${sd}.log" 2>&1
}
for sd in 0 1 2 3 4; do one test $sd & one val $sd & done
wait
$PY - <<'PYEOF' >> "$ST" 2>&1
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
X5={"test":"c5_eval_test_x5g_test_s%d.json","val":"c5_eval_val_x5g_val_s%d.json"}
ok=True; print("\n### S3 경계 섭동 증강 (s3aug) vs X5+gC")
for sp in ("test","val"):
    fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_s3aug_{sp}_s*.json"))
    if not fs: print("  결과 없음"); ok=False; continue
    sds=[int(f.split("_s")[-1][:-5]) for f in fs]
    x=np.array([json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs])
    g=np.array([json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"]["MCC"] for s in sds])
    d=x-g; sd_=d.std(ddof=1) if len(d)>1 else 0; t=d.mean()/(sd_/np.sqrt(len(d))) if sd_>1e-12 else float("nan")
    print(f"  {sp:5s} {x.mean():.4f}±{x.std(ddof=1):.4f}  vs x5g {g.mean():.4f}  Δ{d.mean():+.4f} t{t:+.2f} {int((d>0).sum())}/{len(d)}")
    if d.mean()<=0: ok=False
print(f"판정: {'PASS' if ok else 'FAIL'}")
PYEOF
echo "S3_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$ST"
