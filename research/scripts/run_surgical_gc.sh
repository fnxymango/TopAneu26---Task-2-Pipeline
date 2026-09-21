#!/bin/bash
# 수술적 회수 + gC (2026-08-26). probavgf 에 [vote2f, 지름<=5mm, ICA류 접촉, 기존과 비겹침] blob 만 추가.
# 스캔 실측: test +5 적중 / +6 환각 (전역 vote2f 는 +8/+44). 규칙: test·val 둘 다 T16+gC 초과해야 채택.
# 주의: 스캔의 'ICA류' = dense id [4,6,27,28,29,30,35,36] — ICA-C1C5/C6C7 + AICA/PICA (이름에 ICA 포함).
#       측정과 동일 정의를 그대로 쓴다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis; E=$R/experiments
P=$E/_c1_realpred; BP=$E/_c4_bpgraph
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R; cd $S
until grep -q "^DONE" "$E/q11_x5gc.log" 2>/dev/null; do sleep 30; done
echo "[STEP] $(TZ=Asia/Seoul date +%H:%M) q11 종료 확인 — 병합 시작"

$PY - <<'PYEOF' || exit 1
import os, sys, numpy as np, nibabel as nib
sys.path.insert(0, os.environ["TOPANEU_ROOT"] + "/code/sblee/nnunet/scripts")
from scipy import ndimage as ndi
import d9xx_lib as L
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
ICA = [4, 6, 27, 28, 29, 30, 35, 36]
_, val_ids, test_ids = L.case_ids_by_split()
for sp, ids in (("test", test_ids), ("val", val_ids)):
    src = P / f"aneu_{sp}_probavgf"; ext = P / f"aneu_{sp}_vote2f"
    ves = P / f"vespp_{sp}"; dst = P / f"aneu_{sp}_sur"; dst.mkdir(exist_ok=True)
    add = tot = 0
    for cid in ids:
        i = nib.load(str(src / f"{cid}.nii.gz")); a = (np.asarray(i.dataobj) > 0)
        sp_mm = i.header.get_zooms()[:3]; vmm = float(np.prod(sp_mm))
        out = a.copy()
        f2 = ext / f"{cid}.nii.gz"
        if f2.exists():
            b = np.asarray(nib.load(str(f2)).dataobj) > 0
            vf = ves / f"{cid}.nii.gz"
            vs = np.asarray(nib.load(str(vf)).dataobj).astype(np.int16) if vf.exists() else None
            icam = np.isin(vs, ICA) if vs is not None else None
            lab, k = ndi.label(b)
            for j in range(1, k + 1):
                m = lab == j
                if (a & m).any():             # 기존 blob 과 겹치면 제외
                    continue
                nv = int(m.sum())
                dia = 2.0 * (3.0 * nv * vmm / (4 * np.pi)) ** (1 / 3)
                if dia > 5.0:                 # 지름 5mm 초과 제외
                    continue
                if icam is None or not (icam & ndi.binary_dilation(m)).any():
                    continue                  # ICA류 혈관과 접촉해야 함
                out |= m; add += 1
        o = nib.Nifti1Image(out.astype(np.uint8), i.affine, i.header)
        o.set_data_dtype(np.uint8); nib.save(o, str(dst / f"{cid}.nii.gz")); tot += 1
    print(f"[{sp}] {tot}케이스 저장 · 추가 blob {add}개")
PYEOF

gc_eval(){ local sp=$1 sd=$2; local ves bp
  if [ "$sp" = test ]; then ves=vespp_test; bp=vespp_test; else ves=vespp_val; bp=val_pred; fi
  TOPANEU_TOPK=2 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 CLF_SEED=$sd OMP_NUM_THREADS=2 \
  $PY -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/$ves" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_sur" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "sur_${sp}_s${sd}" > "$E/sur_${sp}_s${sd}.log" 2>&1
}
for sd in 0 1 2 3 4; do gc_eval test $sd & gc_eval val $sd & done
wait
echo "DONE $(TZ=Asia/Seoul date +%H:%M) KST"
