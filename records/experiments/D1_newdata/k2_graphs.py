#!/usr/bin/env python3
"""K2 e2e 재료 — ① 추론 그래프(vespp_test · val_pred)에 좌우 대칭 앵커 보간을 적용한 사본 ② 학습표 bp_mm 교체본.
보간 규칙은 k2_sym.augment 그대로(정중면 거울상 → 그 쪽 ICA-C6-C7 복셀에 붙임 ≤6mm). 추론 쪽 혈관은 그래프를 만든 예측 혈관(vespp_*).
학습표: e11_feat_hyb_ov_NEW 에서 bp_mm 만 k2_sym_train.json 값으로 바꾼다(참조 그래프 all_ref + GT 혈관 — 기준표와 같은 출처).
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
BP = f"{R}/experiments/_c4_bpgraph"; P = f"{R}/experiments/_c1_realpred"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D)
os.environ.setdefault("TOPANEU_ROOT", R)


def one(args):
    src, dst, vdir, cid = args
    import k2_sym as K, d9xx_lib as L
    nm2id = {n: i for i, n in L.vessel_dense_names().items()}
    g = json.load(open(f"{BP}/{src}/{cid}.json"))
    vi = nib.load(f"{P}/{vdir}/{cid}.nii.gz"); ves = np.asanyarray(vi.dataobj); sp = np.array(vi.header.get_zooms()[:3], float)
    added, st = K.augment(g.get("nodes", []), ves, sp, nm2id)
    g["nodes"] = g.get("nodes", []) + added
    g["n_nodes"] = len(g["nodes"]); g["sym_added"] = len(added)
    json.dump(g, open(f"{BP}/{dst}/{cid}.json", "w"))
    return st


def main():
    import multiprocessing as mp
    S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]
    jobs = []
    for sp, src, vdir in (("test", "vespp_test", "vespp_test"), ("val", "val_pred", "vespp_val")):
        dst = f"{src}_sym"; os.makedirs(f"{BP}/{dst}", exist_ok=True)
        for c in S[sp]:
            if os.path.exists(f"{BP}/{src}/{c}.json") and os.path.exists(f"{P}/{vdir}/{c}.nii.gz"):
                jobs.append((src, dst, vdir, c))
    tot = collections.Counter()
    with mp.Pool(8) as p:
        for st in p.map(one, jobs):
            tot.update(st)
    print(f"추론 그래프 보간 {len(jobs)}케이스 · {dict(sorted(tot.items()))}")
    rows = json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json"))
    new = json.load(open(f"{D}/k2_sym_train.json"))
    n = 0
    for r in rows:
        k = f"{r['case']}|{r['lesion_mask_idx']}"
        if r.get("gt_loc") and k in new:
            n += new[k] != r["bp_mm"]; r["bp_mm"] = new[k]
    json.dump(rows, open(f"{A}/k2_feat_hyb_sym.json", "w"))
    base = json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json"))
    assert len(base) == len(rows) and all(
        {k: v for k, v in a.items() if k != "bp_mm"} == {k: v for k, v in b.items() if k != "bp_mm"} for a, b in zip(base, rows))
    print(f"학습표 k2_feat_hyb_sym.json · {len(rows)}행 · bp_mm 바뀐 행 {n} · 나머지 필드 기준표와 동일 확인")


if __name__ == "__main__":
    main()
