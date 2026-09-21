#!/usr/bin/env python
"""S1 좌우 하드제약 + S2 학습 라벨잡음 스크리닝 (2026-08-27).

S1: 병변 중심의 정중선 거리 vs 예측 클래스의 좌우. 명백 편측(>=8mm) 병변에 반대쪽 예측이 있으면
    하드 제약으로 공짜 교정. 정중선 = 케이스 혈관마스크에서 BA(1)+Acom(10) 복셀의 x 중앙값,
    없으면 L/R ICA-C6-C7(6,4) 중심의 중점.
S2: 292 OOF 에서 GT 와 다르게 높은 확신으로 예측된 학습병변 = 의심 라벨. 빼고 재학습해 test/val 병변수준.
"""
import json, os, sys, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

ST = np.ones((3, 3, 3), bool)
A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
NONE = "__none__"; BETA, TAU = 0.5, 0.5
train_ids, val_ids, test_ids = L.case_ids_by_split()

def side_of(c):
    m = re.match(r"^([LR])-", c); return m.group(1) if m else "M"

# ---------- S1 ----------
def midline_and_centroids(sp, adir, vdir):
    """case -> (midline_x_mm, {lesion_idx: (cx_mm, ...)})  x = 배열 0축이 아니라 RAS x 로 환산"""
    out = {}
    for cid in (test_ids if sp == "test" else val_ids if sp == "val" else train_ids):
        fa = P / adir / f"{cid}.nii.gz"; fv = P / vdir / f"{cid}.nii.gz"
        if not fa.exists() or not fv.exists(): continue
        ia = nib.load(str(fa)); a = np.asanyarray(ia.dataobj) > 0
        if not a.any(): continue
        v = np.asanyarray(nib.load(str(fv)).dataobj).astype(np.int16)
        aff = ia.affine
        def to_x(idx):  # voxel idx (N,3) -> world x
            h = np.c_[idx, np.ones(len(idx))]; return (h @ aff.T)[:, 0]
        mid = np.argwhere(np.isin(v, [1, 10]))
        if len(mid) >= 20:
            mx = float(np.median(to_x(mid)))
        else:
            l = np.argwhere(v == 6); r = np.argwhere(v == 4)
            if len(l) == 0 or len(r) == 0: continue
            mx = float((to_x(l).mean() + to_x(r).mean()) / 2)
        # 좌우 부호: L-ICA 가 +x 인지 -x 인지 케이스별로 정한다
        l = np.argwhere(v == 6); r = np.argwhere(v == 4)
        if len(l) == 0 or len(r) == 0: continue
        lsign = 1.0 if to_x(l).mean() > to_x(r).mean() else -1.0
        lab, k = ndi.label(a, structure=ST)
        cents = {}
        for j in range(1, k + 1):
            idx = np.argwhere(lab == j); cents[j] = (float(to_x(idx).mean()) - mx) * lsign  # +면 L 쪽
        out[cid] = cents
    return out

print("[S1] 좌우 하드제약 스크리닝", flush=True)
for sp, adir, vdir in (("trainoof", "aneu_train_ooff", "vespp_train"),
                       ("test", "aneu_test_P55ff", "vespp_test"), ("val", "aneu_val_P55ff", "vespp_val")):
    # 주의: proba_{sp} npz 는 probavgf(T16 검출) 기준 병변 색인이다 → 같은 검출본으로 중심 계산
    adir_p = {"trainoof": "aneu_train_ooff", "test": "aneu_test_probavgf", "val": "aneu_val_probavgf"}[sp]
    geo = midline_and_centroids(sp, adir_p, vdir)
    fs = sorted((A).glob(f"proba_{sp}_s*.npz"))
    tot = wrong_side = fixable = fix_correct = clear = 0
    for f in fs:
        g = np.load(f, allow_pickle=True); Pm, cls, tr, cs, ls = g["P"], list(g["classes"]), list(g["truth"]), g["case"], g["lesion"]
        CI = {c: i for i, c in enumerate(cls)}
        for i in range(len(Pm)):
            if cs[i] not in geo or int(ls[i]) not in geo[cs[i]]: continue
            d = geo[cs[i]][int(ls[i])]; pred = cls[int(Pm[i].argmax())]; t = tr[i]
            tot += 1
            if abs(d) < 8.0: continue
            clear += 1
            gside = "L" if d > 0 else "R"
            if side_of(pred) not in ("M", gside):
                wrong_side += 1
                # 제약 적용: 같은쪽/정중 클래스 중 최고
                ok = [j for j, c in enumerate(cls) if side_of(c) in ("M", gside)]
                newp = cls[ok[int(np.argmax(Pm[i][ok]))]]
                if t:
                    fixable += 1
                    fix_correct += int(newp == t)
    n = len(fs)
    print(f"  {sp:9s} 병변 {tot/n:.0f}/시드 · 편측(≥8mm) {clear/n:.0f} · 반대쪽 예측 {wrong_side/n:.1f}"
          f" · 그중 GT있음 {fixable/n:.1f} → 제약 후 정답 {fix_correct/n:.1f}", flush=True)

# ---------- S2 ----------
print("\n[S2] 학습 라벨잡음 스크리닝", flush=True)
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym, ct = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
prior = collections.Counter(list(yt) + list(ym))
# OOF 확신 집계 (5시드 평균) → (case, truth) 별 P[truth], P[pred], pred
acc = collections.defaultdict(list)
for f in sorted(A.glob("proba_trainoof_s*.npz")):
    g = np.load(f, allow_pickle=True); Pm, cls, tr, cs = g["P"], list(g["classes"]), list(g["truth"]), g["case"]
    CI = {c: i for i, c in enumerate(cls)}
    for i in range(len(Pm)):
        if not tr[i] or tr[i] not in CI: continue
        j = int(Pm[i].argmax()); acc[(cs[i], tr[i])].append((float(Pm[i][CI[tr[i]]]), float(Pm[i][j]), cls[j]))
sus = {}
for k, v in acc.items():
    pt = np.mean([a for a, _, _ in v]); pp = np.mean([b for _, b, _ in v])
    preds = collections.Counter(c for _, _, c in v).most_common(1)[0]
    if pt < 0.10 and pp > 0.50 and preds[1] >= 4:
        sus[k] = (pt, pp, preds[0])
print(f"  의심 라벨 {len(sus)}개 / OOF 대응 학습병변 {len(acc)}개  (기준: P[GT]<0.10 & P[pred]>0.50 & 4/5시드 일치)")
for (c, t), (pt, pp, pr) in sorted(sus.items(), key=lambda x: x[1][1], reverse=True)[:12]:
    print(f"    {c[-10:]}  GT {t:<28} → OOF {pr:<28} P[GT] {pt:.2f}  P[pred] {pp:.2f}")
mask_drop = np.array([(c, y) not in sus for c, y in zip(ct, yt)])
print(f"  제거 대상 학습행 {int((~mask_drop).sum())}/{len(yt)}")

def fit(mask, seed):
    X = np.concatenate([Xt[mask], Xm[mask]]); y = np.concatenate([yt[mask], ym[mask]])
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=4).fit(X, y)
    pri = np.array([prior[c] for c in clf.classes_], dtype=float); return clf, pri
def predict(clf, pri, X):
    Pp = clf.predict_proba(X); hi = Pp.max(axis=1) >= TAU
    Pc = Pp.copy(); Pc[~hi] = Pc[~hi] / (pri ** BETA)
    return [str(clf.classes_[i]) for i in Pc.argmax(axis=1)]
def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]; present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))
for tag, lab in (("test", "test 83"), ("val", "val 42")):
    f = np.load(A / f"feat_{tag}.npz", allow_pickle=True); Xe, tr = f["X"], list(f["truth"])
    res = {False: [], True: []}
    for use in (False, True):
        for sd in range(5):
            clf, pri = fit(mask_drop if use else np.ones(len(yt), bool), sd)
            res[use].append(score(tr, predict(clf, pri, Xe)))
    b = np.array(res[False]).mean(axis=0); m = np.array(res[True]).mean(axis=0)
    d = np.array(res[True])[:, 2] - np.array(res[False])[:, 2]
    print(f"  [{lab}] 기준 MCC {b[2]:.4f}  → 의심라벨 제거 {m[2]:.4f}  Δ{m[2]-b[2]:+.4f}  {int((d>0).sum())}/5   (macroRec {b[1]:.3f}→{m[1]:.3f})", flush=True)
print("완료")
