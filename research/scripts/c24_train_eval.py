"""C24-b — 크롭 분류기: 게이트 검증 → 학습 → 기하 RF 와 앙상블.

C14 가 2회 실패한 원인(얼린 인코더 / 좌표 미검증)을 막기 위해 **게이트를 먼저 통과**시킨다.
  게이트1  크롭에 병변이 실제로 들어갔는가 (build 단계의 lesion_in_crop)
  게이트2  영상 채널만으로 랜덤(1/43≈0.023)을 유의하게 넘는가
게이트2를 못 넘으면 3D 학습은 건너뛰고 **2D 다중뷰 MIP 대안**으로 넘어간다
(파라미터가 훨씬 적고, 실제 영상의학과가 혈관조영을 읽는 방식이라 소규모 데이터에 강하다).

최종 목표는 크롭 분류기 단독 성능이 아니라 **기하 RF 와의 앙상블**이다.
두 분류기의 실패 패턴이 다르면(기하는 분지 미검출에서, 영상은 대비 불량에서) 상보적이다.

사용: python c24_train_eval.py --crop-dir <train크롭> --feat <c10_feat_train.json>
"""
import argparse, collections, json, os
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8

RNG = np.random.default_rng(20260817)


def load_crops(crop_dir):
    meta = json.load(open(Path(crop_dir) / "meta.json"))
    meta = [m for m in meta if m.get("gt_loc")]
    X = np.stack([np.load(Path(crop_dir) / f"{m['key']}.npz")["x"] for m in meta])
    return meta, X.astype(np.float32)


def mip3(X):
    """(N,C,D,H,W) -> (N, C*3, H, W) 다중뷰 최대투영. 3D 대비 파라미터가 훨씬 적다."""
    return np.concatenate([X.max(axis=2), X.max(axis=3), X.max(axis=4)], axis=1)


def gate2(X, y, groups, channels):
    """영상 채널만으로 랜덤을 넘는지 — 가벼운 MIP+RF 로 빠르게 확인."""
    F = mip3(X[:, channels]).reshape(len(X), -1)
    step = max(1, F.shape[1] // 3000)
    F = F[:, ::step]
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=5).split(F, y, groups):
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(F[tr], y[tr])
        pred[te] = clf.predict(F[te])
    return float(np.mean(pred == y))


def train_cnn(X, y, groups, epochs=120, mip=False):
    """작은 3D(또는 2D-MIP) CNN. 268샘플이라 얕게 + 강한 정규화 + 미러 증강."""
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = sorted(set(y)); c2i = {c: i for i, c in enumerate(classes)}
    yi = np.array([c2i[v] for v in y])
    mir = np.array([c2i.get(C5.mirror_name(v), c2i[v]) for v in y])
    proba = np.zeros((len(y), len(classes)), dtype=np.float32)

    def make_net(cin, is2d):
        Conv, Pool, BN = (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.BatchNorm2d) if is2d \
            else (nn.Conv3d, nn.AdaptiveAvgPool3d, nn.BatchNorm3d)
        L_ = []
        c = cin
        for co in (16, 32, 64):
            L_ += [Conv(c, co, 3, stride=2, padding=1), BN(co), nn.LeakyReLU(0.01, True)]
            c = co
        L_ += [Pool(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(c, len(classes))]
        return nn.Sequential(*L_)

    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        Xtr = mip3(X[tr]) if mip else X[tr]
        Xte = mip3(X[te]) if mip else X[te]
        net = make_net(Xtr.shape[1], mip).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        cnt = collections.Counter(yi[tr])
        w = torch.tensor([1.0 / cnt.get(i, 1) for i in range(len(classes))],
                         dtype=torch.float32, device=dev)
        w = w / w.mean()
        lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
        xb_all = torch.tensor(Xtr, device=dev)
        yb_all = torch.tensor(yi[tr], device=dev)
        mb_all = torch.tensor(mir[tr], device=dev)
        n = len(tr)
        net.train()
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, 8):
                idx = perm[s:s + 8]
                xb, yb = xb_all[idx], yb_all[idx]
                if RNG.random() < 0.5:                    # 좌우 미러 + 라벨 교체
                    xb = torch.flip(xb, dims=[-1]); yb = mb_all[idx]
                xb = xb * (1 + 0.1 * torch.randn(len(idx), 1, 1, 1, device=dev
                                                 ).view(-1, *([1] * (xb.dim() - 1))))
                opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            p = torch.softmax(net(torch.tensor(Xte, device=dev)), dim=1).cpu().numpy()
        proba[te] = p
    return classes, proba


def geo_probs(feat_path, keys, folds=5):
    """같은 fold 분할로 기하 RF 의 확률행렬을 만든다(앙상블용)."""
    C5.USE_POS = True
    rows = [r for r in json.load(open(feat_path)) if r.get("gt_loc")]
    idx = {f"{r['case']}__{r['lesion_mask_idx']}": i for i, r in enumerate(rows)}
    sel = [idx.get(k) for k in keys]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    X = np.array([C5.row_to_vec(r, ves_axis) for r in rows])
    y = np.array([r["gt_loc"] for r in rows])
    pm = C8.patient_map()
    g = np.array([pm.get(r["case"], r["case"]) for r in rows])
    classes = sorted(set(y))
    P = np.zeros((len(rows), len(classes)), dtype=np.float32)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        Xa, ya = [], []
        for i in tr:
            Xa.append(X[i]); ya.append(y[i])
            Xa.append(C5.row_to_vec(rows[i], ves_axis, mirror=True))
            ya.append(C5.mirror_name(y[i]))
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(np.array(Xa), np.array(ya))
        pr = clf.predict_proba(X[te])
        cmap = {c: j for j, c in enumerate(clf.classes_)}
        for k, c in enumerate(classes):
            if c in cmap:
                P[te, k] = pr[:, cmap[c]]
    return classes, P, sel


def score(y, pred):
    t1 = float(np.mean(y == pred))
    mr, _ = C8.macro_recall(y, pred)
    ica = [(a, b) for a, b in zip(y, pred) if C8.group_of(a) == "3"]
    return t1, mr, (float(np.mean([a == b for a, b in ica])) if ica else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-dir", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args()

    meta, X = load_crops(a.crop_dir)
    y = np.array([m["gt_loc"] for m in meta])
    pm = C8.patient_map()
    groups = np.array([pm.get(m["case"], m["case"]) for m in meta])
    keys = [f"{m['case']}__{m['lesion_idx']}" for m in meta]
    print(f"[c24] 크롭 {len(meta)} · 채널 {X.shape[1]} · 크기 {X.shape[2:]}\n")

    # ── 게이트1: 크롭에 병변이 들어갔는가 ──
    empty = sum(1 for m in meta if m["lesion_in_crop"] < 1)
    print(f"[게이트1] 병변이 크롭에 안 잡힌 것 {empty}/{len(meta)}")
    if empty > len(meta) * 0.05:
        print("  ✗ 실패 — 좌표 매핑을 먼저 고쳐야 함. 중단."); return
    print("  ✓ 통과\n")

    # ── 게이트2: 영상 채널 단독이 랜덤을 넘는가 ──
    rand = 1.0 / len(set(y))
    acc_img = gate2(X, y, groups, [0])
    acc_geo = gate2(X, y, groups, [1, 2])
    print(f"[게이트2] 랜덤 {rand:.3f} | 영상채널만 {acc_img:.3f} | 혈관채널만 {acc_geo:.3f}")
    use_mip = acc_img < rand * 3
    print("  ✓ 영상에 신호 있음" if not use_mip else
          "  ⚠ 영상 신호 약함 — 3D 대신 2D-MIP 대안으로 진행")
    print()

    # ── 학습 ──
    print(f"[학습] {'2D-MIP' if use_mip else '3D'} CNN, {a.epochs} epoch, 환자단위 5-fold")
    classes_c, P_crop = train_cnn(X, y, groups, epochs=a.epochs, mip=use_mip)
    pred_c = np.array(classes_c)[np.argmax(P_crop, axis=1)]
    t1, mr, ic = score(y, pred_c)
    print(f"  크롭 분류기 단독: top-1 {t1:.3f} · macro-recall {mr:.3f} · ICA {ic:.3f}\n")

    # ── 앙상블 ──
    classes_g, P_geo_all, sel = geo_probs(a.feat, keys)
    ok = [i for i, s in enumerate(sel) if s is not None]
    if len(ok) < len(meta) * 0.8:
        print(f"[앙상블] 기하 피처 매칭 {len(ok)}/{len(meta)} — 건너뜀"); return
    P_geo = P_geo_all[[sel[i] for i in ok]]
    cmap = {c: j for j, c in enumerate(classes_g)}
    Pc = np.zeros_like(P_geo)
    for j, c in enumerate(classes_c):
        if c in cmap:
            Pc[:, cmap[c]] = P_crop[ok, j]
    yo = y[ok]
    prior = collections.Counter(yo)
    pri = np.array([max(prior.get(c, 1), 1) for c in classes_g], dtype=float)

    print(f"{'w(크롭)':>8}{'top-1':>9}{'macroRec':>10}{'ICA':>8}")
    best = None
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        P = (w * Pc + (1 - w) * P_geo)
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-9) / pri   # beta=1.0
        pr = np.array(classes_g)[np.argmax(P, axis=1)]
        t1, mr, ic = score(yo, pr)
        print(f"{w:>8.1f}{t1:>9.3f}{mr:>10.3f}{ic:>8.3f}")
        if best is None or mr > best[1]:
            best = (w, mr, t1, ic)
    print(f"\n[최적] w={best[0]:.1f} · macro-recall {best[1]:.3f} "
          f"(기하 단독 w=0 대비 {best[1]/max(score(yo, np.array(classes_g)[np.argmax(P_geo/pri,axis=1)])[1],1e-9)-1:+.1%})")

    out = Path(a.feat).parent / "c24_report.json"
    json.dump({"n_crops": len(meta), "gate_img_acc": acc_img, "gate_geo_acc": acc_geo,
               "random": rand, "used_mip": bool(use_mip),
               "crop_only": {"top1": float(np.mean(pred_c == y))},
               "best_w": best[0], "best_macro_recall": best[1]},
              open(out, "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(Path(a.feat).parent / "c24_crop_probs.npz",
                        classes=np.array(classes_c, dtype=object), P=P_crop,
                        keys=np.array(keys, dtype=object))
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
