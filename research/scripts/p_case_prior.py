#!/usr/bin/env python
"""N2 — 케이스 수준 혈관 변이 prior (2026-08-26).

지금까지 전부 병변 수준이었다. Willis环 변이(A1 저형성·태아형 PCA·Acom 부재 등)는
**케이스 단위로** 등장 가능한 클래스를 바꾼다. 예측 혈관 마스크의 36분절 부피(로그)를
케이스 피처로 써서 P(클래스 존재|케이스) 를 학습하고, 병변 확률에 곱한다:
  P'(c|b) ∝ P(c|b) · P(c present|case)^w      w 는 292 OOF 로 고르고 test·val 로 판정.

학습라벨: 292 케이스의 GT 병변 클래스 존재(케이스별 멀티라벨). 혈관은 vespp_train(in-sample
이지만 부피 같은 거시 피처는 안정적) — test/val 은 각자 예측 혈관.
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
NONE = "__none__"
train_ids, val_ids, test_ids = L.case_ids_by_split()

VOL = A / "case_vessel_vols.npz"
if VOL.exists():
    z = np.load(VOL, allow_pickle=True); vols = dict(zip(z["case"], z["V"]))
else:
    vols = {}
    for sp, ids, vd in (("train", train_ids, "vespp_train"), ("val", val_ids, "vespp_val"),
                        ("test", test_ids, "vespp_test")):
        for i, cid in enumerate(ids, 1):
            f = P / vd / f"{cid}.nii.gz"
            if not f.exists(): continue
            im = nib.load(str(f)); a = np.asanyarray(im.dataobj).astype(np.int16)
            vmm = float(np.prod(im.header.get_zooms()[:3]))
            cnt = np.bincount(a.ravel(), minlength=37)[1:37]
            vols[cid] = np.log1p(cnt * vmm)
        print(f"  부피 {sp} 완료", flush=True)
    cs = sorted(vols)
    np.savez_compressed(VOL, case=np.array(cs, dtype=object), V=np.stack([vols[c] for c in cs]))
    print("[저장] case_vessel_vols.npz", flush=True)

# 케이스 존재 라벨 (train 292 GT)
rows = json.load(open(A / "e11_feat_hyb_ov.json"))
pres = collections.defaultdict(set)
for r in rows: pres[r["case"]].add(r["gt_loc"])
ft = np.load(A / "feat_train.npz", allow_pickle=True)
CLASSES = sorted(set(list(ft["y"])) | set(list(ft["ym"]))); CI = {c: i for i, c in enumerate(CLASSES)}

def fit_prior(ids):
    X = np.stack([vols[c] for c in ids if c in vols])
    ids2 = [c for c in ids if c in vols]
    models = {}
    for c in CLASSES:
        y = np.array([c in pres[i] for i in ids2], dtype=int)
        if y.sum() < 3 or y.sum() > len(y) - 3:
            models[c] = float(y.mean()) if len(y) else 0.0; continue
        models[c] = LogisticRegression(C=0.3, max_iter=500).fit(X, y)
    return models

def prior_vec(models, cid):
    v = vols.get(cid)
    out = np.full(len(CLASSES), 0.5)
    if v is None: return out
    for c, m in models.items():
        out[CI[c]] = m if isinstance(m, float) else float(m.predict_proba(v[None])[0, 1])
    return np.clip(out, 1e-3, 1 - 1e-3)

def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))

WS = [0.0, 0.25, 0.5, 1.0]
for tag, lab in (("trainoof", "292 OOF"), ("test", "test 83"), ("val", "val 42")):
    res = collections.defaultdict(list)
    for sd in range(5):
        g = np.load(A / f"proba_{tag}_s{sd}.npz", allow_pickle=True)
        Pm, cls, tr, cs = g["P"], list(g["classes"]), list(g["truth"]), g["case"]
        assert cls == CLASSES
        if tag == "trainoof":
            pv = {}
            for fd in json.load(open(SP720)):
                va = set(fd["val"])
                if not any(c in va for c in cs): continue
                mods = fit_prior([c for c in train_ids if c not in va])
                for c in set(cs) & va: pv[c] = prior_vec(mods, c)
        else:
            mods = fit_prior(train_ids)
            pv = {c: prior_vec(mods, c) for c in set(cs)}
        for w in WS:
            Q = Pm * np.stack([pv[c] for c in cs]) ** w
            res[w].append(score(tr, [CLASSES[i] for i in Q.argmax(axis=1)]))
    b = np.array(res[0.0]).mean(axis=0)
    print(f"\n[{lab}]  (w=0 현행)")
    for w in WS:
        m = np.array(res[w]).mean(axis=0)
        d = np.array(res[w])[:, 2] - np.array(res[0.0])[:, 2]
        ex = "" if w == 0 else f"   ΔMCC {m[2]-b[2]:+.4f}  {int((d>0).sum())}/5"
        print(f"  w={w:<5} top1 {m[0]:.3f}  macroRec {m[1]:.3f}  MCC {m[2]:.4f}{ex}", flush=True)
print("\n완료")
