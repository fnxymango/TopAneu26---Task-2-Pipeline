#!/bin/bash
# F3 검출확신 게이트 (2026-08-26 밤): P55ff blob 중 검출확신 낮은 것 삭제 -> e2e.
# 근거(F2): 환각분리 AUC test 0.865 / val 0.814. 규칙(사전고정): 문턱 {0.85,0.90} 각각
#   X5+gC(시드0-4: test 0.3972/val 0.4725) 대비 test·val 둘 다 + 여야 채택. 둘 다 통과시 Δ합 큰 쪽.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
ST=$E/chain_status.md
until grep -q "F2_DONE" "$ST" 2>/dev/null; do sleep 60; done
echo "[$(TZ=Asia/Seoul date +%H:%M)] F3 검출확신 게이트 시작" >> "$ST"

$PY - <<'PYEOF' >> "$ST" 2>&1 || exit 1
import os, sys, numpy as np, nibabel as nib
sys.path.insert(0, os.environ["TOPANEU_ROOT"] + "/code/sblee/nnunet/scripts")
from scipy import ndimage as ndi
import d9xx_lib as L
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
_, val_ids, test_ids = L.case_ids_by_split()
for th in (0.85, 0.90):
    tag = f"dc{int(th*100)}"
    for sp, ids in (("test", test_ids), ("val", val_ids)):
        dst = P / f"aneu_{sp}_P55ff_{tag}"; dst.mkdir(exist_ok=True)
        kept = cut = 0
        for cid in ids:
            fb = P / f"aneu_{sp}_P55ff" / f"{cid}.nii.gz"
            fp_ = P / f"aneu_{sp}_P55prob" / f"{cid}.npz"
            i = nib.load(str(fb)); a = np.asarray(i.dataobj) > 0
            out = a.copy()
            if a.any() and fp_.exists():
                pr = np.load(fp_)["probabilities"]
                pa = pr[2] if pr.shape[0] > 2 else pr[-1]
                if pa.shape != a.shape: pa = np.transpose(pa, (2, 1, 0))
                if pa.shape == a.shape:
                    lab, k = ndi.label(a)
                    for j in range(1, k + 1):
                        m = lab == j
                        if float(pa[m].mean()) < th: out[m] = False; cut += 1
                        else: kept += 1
            o = nib.Nifti1Image(out.astype(np.uint8), i.affine, i.header)
            o.set_data_dtype(np.uint8); nib.save(o, str(dst / f"{cid}.nii.gz"))
        print(f"  [{tag} {sp}] 유지 {kept} · 삭제 {cut}")
PYEOF

one(){ local sp=$1 sd=$2 tag=$3; local ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_P55ff_${tag}" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tag}_${sp}_s${sd}" > "$E/${tag}_${sp}_s${sd}.log" 2>&1
}
for tag in dc85 dc90; do
  for sd in 0 1 2 3 4; do one test $sd $tag & one val $sd $tag & done
  wait
done
$PY - <<'PYEOF' >> "$ST" 2>&1
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
X5={"test":"c5_eval_test_x5g_test_s%d.json","val":"c5_eval_val_x5g_val_s%d.json"}
for tag in ("dc85","dc90"):
    ok=True
    print(f"\n### F3 검출확신 게이트 ({tag})")
    for sp in ("test","val"):
        fs=sorted(glob.glob(f"{A}/c5_eval_{sp}_{tag}_{sp}_s*.json"))
        if not fs: print("  결과 없음"); ok=False; continue
        sds=[int(f.split("_s")[-1][:-5]) for f in fs]
        x=np.array([json.load(open(f))["adjusted_div_present"]["MCC"] for f in fs])
        g=np.array([json.load(open(A+"/"+X5[sp]%s))["adjusted_div_present"]["MCC"] for s in sds])
        d=x-g; sd_=d.std(ddof=1) if len(d)>1 else 0
        t=d.mean()/(sd_/np.sqrt(len(d))) if sd_>1e-12 else float("nan")
        print(f"  {sp:5s} {x.mean():.4f}±{x.std(ddof=1):.4f}  vs x5g {g.mean():.4f}  Δ{d.mean():+.4f} t{t:+.2f} {int((d>0).sum())}/{len(d)}")
        if d.mean()<=0: ok=False
    print(f"판정: {'PASS' if ok else 'FAIL'}")
    open(A+f"/chain_{tag}.verdict","w").write("PASS" if ok else "FAIL")
PYEOF
echo "F3_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$ST"

# F4: T16+gC 시드 5~9 (10v10 기록 완성 — 판정 재개봉 아님, 분산 기록용)
oneT(){ local sp=$1 sd=$2; local ves bp an
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; an=aneu_test_probavgf
  else ves=vespp_val; bp=val_pred; an=aneu_val_probavgf; fi
  [ -f "$A/c5_eval_${sp}_gC_${sp}_s${sd}.json" ] && return 0
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$an" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "gC_${sp}_s${sd}" > "$E/gC10_${sp}_s${sd}.log" 2>&1
}
for sd in 5 6 7 8 9; do oneT test $sd & oneT val $sd & done
wait
$PY - <<'PYEOF' >> "$ST" 2>&1
import json,glob,numpy as np
A="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
print("\n### F4 10시드 기록 — X5+gC vs T16+gC (짝지은 10시드)")
for sp in ("test","val"):
    fx=sorted(glob.glob(f"{A}/c5_eval_{sp}_x5g_{sp}_s*.json"))
    fg=sorted(glob.glob(f"{A}/c5_eval_{sp}_gC_{sp}_s*.json"))
    sx={int(f.split("_s")[-1][:-5]):f for f in fx}; sg={int(f.split("_s")[-1][:-5]):f for f in fg}
    ks=sorted(set(sx)&set(sg))
    x=np.array([json.load(open(sx[k]))["adjusted_div_present"]["MCC"] for k in ks])
    g=np.array([json.load(open(sg[k]))["adjusted_div_present"]["MCC"] for k in ks])
    d=x-g; t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"  {sp:5s} n={len(ks)}  X5+gC {x.mean():.4f}±{x.std(ddof=1):.4f}  T16+gC {g.mean():.4f}±{g.std(ddof=1):.4f}  Δ{d.mean():+.4f} t{t:+.2f} {int((d>0).sum())}/{len(d)}")
PYEOF
echo "F4_DONE $(TZ=Asia/Seoul date +%H:%M) KST" >> "$ST"
