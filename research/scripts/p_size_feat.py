#!/usr/bin/env python
"""④ 크기 피처 스크리닝 (2026-08-26).

현행 106+6차원에 병변 크기가 없다. 그런데 미검출·클래스 분석에서 크기와 클래스가
강하게 엮인다(ICA=작음, MCA 분지=큼). n_vox 는 이미 추출돼 있고(row dict) X 에만 빠졌다.

추가 컬럼 2개: log10(부피 mm^3), 등가구 지름(mm). 미러 행은 같은 값.
병변수준 5시드: 292 OOF(폴드 재적합) / test / val. 셋 다 + 여야 공식 e2e 로 넘긴다.
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

ST = np.ones((3, 3, 3), bool)
A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
NONE = "__none__"
BETA, TAU = 0.5, 0.5

spc = {}
def spacing(cid):
    if cid not in spc:
        h = nib.load(str(L.DATA / "location_masks" / f"{cid}.nii.gz")).header
        spc[cid] = float(np.prod(h.get_zooms()[:3]))
    return spc[cid]

def size_cols(nvox, vmm):
    mm3 = max(nvox * vmm, 1e-6)
    dia = 2.0 * (3.0 * mm3 / (4 * np.pi)) ** (1 / 3)
    return [np.log10(mm3), dia]

# 학습: e11 rows 의 n_vox
rows = json.load(open(A / "e11_feat_hyb_ov.json"))
St = np.array([size_cols(r["n_vox"], spacing(r["case"])) for r in rows])
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym, ct = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
assert len(St) == len(Xt)
CLASSES = sorted(set(list(yt) + list(ym)))
prior = collections.Counter(list(yt) + list(ym))
print(f"학습 {len(Xt)} · 크기 중앙값 dia {np.median(St[:,1]):.2f}mm", flush=True)

# 평가쪽: pred 마스크 라벨링 재현으로 lesion idx -> nvox
def eval_sizes(tag, adir):
    f = np.load(A / f"feat_{tag}.npz", allow_pickle=True)
    out = np.zeros((len(f["X"]), 2))
    cache = {}
    for i, (cid, lid) in enumerate(zip(f["case"], f["lesion"])):
        if cid not in cache:
            im = nib.load(str(P / adir / f"{cid}.nii.gz"))
            lab, _ = ndi.label(np.asanyarray(im.dataobj) > 0, structure=ST)
            cache[cid] = (lab, float(np.prod(im.header.get_zooms()[:3])))
        lab, vmm = cache[cid]
        out[i] = size_cols(int((lab == lid).sum()), vmm)
    return f["X"], list(f["truth"]), f["case"], out

def fit(mask, seed, use_size, St_):
    X = np.concatenate([Xt[mask], Xm[mask]]); y = np.concatenate([yt[mask], ym[mask]])
    if use_size:
        S2 = np.concatenate([St_[mask], St_[mask]])
        X = np.hstack([X, S2])
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                 class_weight="balanced", random_state=seed, n_jobs=4).fit(X, y)
    pri = np.array([prior[c] for c in clf.classes_], dtype=float)
    return clf, pri

def predict(clf, pri, X):
    Pp = clf.predict_proba(X)
    hi = Pp.max(axis=1) >= TAU
    Pc = Pp.copy(); Pc[~hi] = Pc[~hi] / (pri ** BETA)
    return [str(clf.classes_[i]) for i in Pc.argmax(axis=1)]

def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))

DATA = {}
DATA["test"] = eval_sizes("test", "aneu_test_probavgf")
DATA["val"] = eval_sizes("val", "aneu_val_probavgf")
DATA["trainoof"] = eval_sizes("trainoof", "aneu_train_ooff")
print("평가 크기 추출 완료", flush=True)

for tag in ("trainoof", "test", "val"):
    Xe, tr, cs, Se = DATA[tag]
    res = {False: [], True: []}
    for use in (False, True):
        for sd in range(5):
            pred = np.empty(len(Xe), dtype=object)
            if tag == "trainoof":
                for fd in json.load(open(SP720)):
                    va = set(fd["val"]); sel = np.array([c in va for c in cs])
                    if not sel.any(): continue
                    clf, pri = fit(np.array([c not in va for c in ct]), sd, use, St)
                    Xin = np.hstack([Xe[sel], Se[sel]]) if use else Xe[sel]
                    pred[sel] = predict(clf, pri, Xin)
                ok = pred != None
                res[use].append(score([t for t, o in zip(tr, ok) if o], list(pred[ok])))
            else:
                clf, pri = fit(np.ones(len(Xt), bool), sd, use, St)
                Xin = np.hstack([Xe, Se]) if use else Xe
                res[use].append(score(tr, predict(clf, pri, Xin)))
        print(f"  {tag} use_size={use} 완료", flush=True)
    b = np.array(res[False]).mean(axis=0); m = np.array(res[True]).mean(axis=0)
    d = np.array(res[True]) - np.array(res[False])
    lab = {"trainoof": "292 OOF", "test": "test 83", "val": "val 42"}[tag]
    print(f"\n[{lab}] 병변 {len(Xe)}")
    print(f"  기준     top1 {b[0]:.3f}  macroRec {b[1]:.3f}  MCC {b[2]:.4f}")
    print(f"  +크기    top1 {m[0]:.3f}  macroRec {m[1]:.3f}  MCC {m[2]:.4f}"
          f"   ΔMCC {m[2]-b[2]:+.4f}  {int((d[:,2]>0).sum())}/5\n", flush=True)
print("완료")
