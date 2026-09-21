#!/usr/bin/env python
"""Q4 — 크롭 3D CNN, 5폴드 CV (2026-08-26).

C25(크롭 MIP+RF)는 top-1 0.183 으로 기하(0.496)에 크게 못 미쳤고 '크롭만 맞힘'이 12/268
이었다. 크롭 v2 에서 세 가지를 고쳤다:
  (a) 대상 병변 마스크를 채널로 넣는다  (C24 는 중심만 맞추고 대상을 특정 못 함)
  (b) 혈관을 ves/36 연속값 대신 의미 이진채널로 (명목형을 순서형으로 넣던 오류)
  (c) 32mm/64^3 = 0.5mm 등방  (C24 는 0.76mm — Pcom·AChA 지름이 1복셀 미만이었다)

★ 게이트: 같은 폴드에서 RF 대비 '크롭만 맞힘' 이 12/268 을 못 넘으면 중단.
"""
import json, os, sys, collections, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
C = L.TOPANEU_ROOT / "experiments" / "_crops2"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
dev = "cuda"
meta = json.load(open(C / "train" / "meta.json"))
meta = [m for m in meta if m["gt_loc"]]
CLASSES = sorted({m["gt_loc"] for m in meta}); CI = {c: i for i, c in enumerate(CLASSES)}
print(f"크롭 {len(meta)} · 클래스 {len(CLASSES)}", flush=True)
X = np.stack([np.load(C / "train" / f"{m['key']}.npz")["x"] for m in meta]).astype(np.float32)
y = np.array([CI[m["gt_loc"]] for m in meta])
case = np.array([m["case"] for m in meta])
print(f"X {X.shape}  {X.nbytes/1e9:.2f}GB", flush=True)


class Net(nn.Module):
    def __init__(s, nc):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv3d(i, o, 3, 2, 1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.01, True),
                                            nn.Conv3d(o, o, 3, 1, 1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.01, True))
        s.f = nn.Sequential(blk(5, 24), blk(24, 48), blk(48, 96), blk(96, 128))
        s.h = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, nc))
    def forward(s, x): return s.h(s.f(x))


def aug(xb):
    n = xb.shape[0]
    xb = xb.clone()
    xb[:, 0] *= (1 + 0.15 * torch.randn(n, 1, 1, 1, device=xb.device))
    xb[:, 0] += 0.10 * torch.randn(n, 1, 1, 1, device=xb.device)
    xb[:, 0] += 0.05 * torch.randn_like(xb[:, 0])
    for i in range(n):
        sh = [int(t) for t in torch.randint(-4, 5, (3,))]
        xb[i] = torch.roll(xb[i], shifts=sh, dims=(1, 2, 3))
    return xb


def train_fold(tr, va, seed=0, epochs=90):
    torch.manual_seed(seed)
    net = Net(len(CLASSES)).to(dev)
    cnt = np.bincount(y[tr], minlength=len(CLASSES)).astype(float)
    w = torch.tensor(np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0), dtype=torch.float32, device=dev)
    w = w / w.sum() * (cnt > 0).sum()
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=epochs * max(1, len(tr) // 16))
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    Xtr = torch.tensor(X[tr], device=dev); ytr = torch.tensor(y[tr], device=dev)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr) - 15, 16):
            idx = perm[i:i + 16]
            opt.zero_grad()
            out = net(aug(Xtr[idx]))
            l = lossf(out, ytr[idx]); l.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        P = torch.softmax(net(torch.tensor(X[va], device=dev)), 1).cpu().numpy()
    return P, net


folds = json.load(open(SP720))
OOF = np.zeros((len(y), len(CLASSES)), dtype=np.float32); done = np.zeros(len(y), bool)
for k, fd in enumerate(folds):
    va_c = set(fd["val"]); va = np.array([c in va_c for c in case])
    if not va.any(): continue
    tr = np.where(~va)[0]; vai = np.where(va)[0]
    P, _ = train_fold(tr, vai)
    OOF[vai] = P; done[va] = True
    acc = (P.argmax(1) == y[vai]).mean()
    print(f"  fold{k}  학습 {len(tr)} · 검증 {len(vai)} · top-1 {acc:.3f}", flush=True)
np.savez_compressed(A / "crop_cnn_oof.npz", P=OOF, classes=np.array(CLASSES, dtype=object),
                    y=y, case=case, key=np.array([m["key"] for m in meta], dtype=object))

# ---- 게이트: 같은 폴드에서 RF 와 일치표 ----
from sklearn.ensemble import RandomForestClassifier
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xf, Xm, yf, ym, cf = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
key_of = {(m["case"], m["lesion_mask_idx"]): i for i, m in enumerate(meta)}
rf_pred = np.full(len(y), -1)
for fd in folds:
    va_c = set(fd["val"])
    trm = np.array([c not in va_c for c in cf]); vam = ~trm
    if not vam.any(): continue
    Xtr = np.concatenate([Xf[trm], Xm[trm]]); ytr = np.concatenate([yf[trm], ym[trm]])
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                 random_state=0, n_jobs=-1).fit(Xtr, ytr)
    pr = clf.predict(Xf[vam])
    for j, i in enumerate(np.where(vam)[0]):
        rf_pred[i] = CI.get(pr[j], -1)
cnn_pred = OOF.argmax(1)
both = int(((cnn_pred == y) & (rf_pred == y)).sum())
rf_only = int(((cnn_pred != y) & (rf_pred == y)).sum())
cnn_only = int(((cnn_pred == y) & (rf_pred != y)).sum())
neither = int(((cnn_pred != y) & (rf_pred != y)).sum())
print(f"\n[Q4 게이트 · 268 GT 병변 · 같은 5폴드]")
print(f"  RF  top-1 {(rf_pred==y).mean():.3f}   CNN top-1 {(cnn_pred==y).mean():.3f}")
print(f"  일치표  둘다맞음 {both} · 기하만 {rf_only} · 크롭만 {cnn_only} · 둘다틀림 {neither}")
print(f"  오라클 상한 top-1 {(both+rf_only+cnn_only)/len(y):.3f}")
print(f"\n  ★ 게이트: 크롭만 {cnn_only} vs C25 의 12 → "
      f"{'통과 — 혼합 진행' if cnn_only > 12 else '미달 — 중단'}")
