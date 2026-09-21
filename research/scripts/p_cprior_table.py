#!/usr/bin/env python
"""N2 케이스 prior 테이블 — 292 전체로 적합해 test+val 케이스에 적용 (2026-08-26)."""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
z = np.load(A / "case_vessel_vols.npz", allow_pickle=True)
vols = dict(zip(z["case"], z["V"]))
train_ids, val_ids, test_ids = L.case_ids_by_split()
rows = json.load(open(A / "e11_feat_hyb_ov.json"))
pres = collections.defaultdict(set)
for r in rows: pres[r["case"]].add(r["gt_loc"])
ft = np.load(A / "feat_train.npz", allow_pickle=True)
CLASSES = sorted(set(list(ft["y"])) | set(list(ft["ym"])))
ids2 = [c for c in train_ids if c in vols]
X = np.stack([vols[c] for c in ids2])
out = {}
mods = {}
for c in CLASSES:
    y = np.array([c in pres[i] for i in ids2], dtype=int)
    mods[c] = float(y.mean()) if (y.sum() < 3 or y.sum() > len(y) - 3) \
        else LogisticRegression(C=0.3, max_iter=500).fit(X, y)
for cid in list(test_ids) + list(val_ids):
    v = vols.get(cid)
    if v is None: continue
    out[cid] = {c: (m if isinstance(m, float) else float(m.predict_proba(v[None])[0, 1]))
                for c, m in mods.items()}
json.dump(out, open(A / "case_prior.json", "w"))
print(f"[저장] case_prior.json  케이스 {len(out)}")
