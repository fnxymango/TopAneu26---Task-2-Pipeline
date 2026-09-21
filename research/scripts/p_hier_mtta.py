#!/usr/bin/env python
"""M1 계층 argmax + M2 좌우 TTA 병변수준 스크리닝 (2026-08-26).

M1: 라벨틀림의 58.5% 가 같은 혈관 인접분절이다. flat argmax 는 정답 혈관의 확률이
    인접분절로 쪼개질 때 진다. [가족(대분류×좌우) 합산 -> 가족 선택 -> 가족 내 argmax].
M2: 병변 피처를 미러링해 예측하고 클래스명을 되뒤집어 확률 평균. 좌우 오류 35.8% 공략.
둘 다 학습 불변·추론만 바뀜. 292 OOF / test / val 셋 다 + 여야 e2e 진행.
"""
import json, os, sys, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L, c5_location_v2 as C5

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
NONE = "__none__"; BETA, TAU = 0.5, 0.5
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym, ct = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
CLASSES = sorted(set(list(yt) + list(ym))); CI = {c: i for i, c in enumerate(CLASSES)}
prior = collections.Counter(list(yt) + list(ym))
MIRROR_IDX = np.array([CI.get(C5.mirror_name(c), CI[c]) for c in CLASSES])

def fam(c):
    m = re.match(r"^([LR])-(\d+)\.", c)
    if m: return f"{m.group(1)}{m.group(2)}"
    m = re.match(r"^(\d+)\.", c)
    return f"M{m.group(1)}" if m else c
FAM = [fam(c) for c in CLASSES]
FAMS = sorted(set(FAM)); FI = {f: i for i, f in enumerate(FAMS)}
FMAT = np.zeros((len(CLASSES), len(FAMS))); 
for i, f in enumerate(FAM): FMAT[i, FI[f]] = 1

def fit(mask, seed):
    X = np.concatenate([Xt[mask], Xm[mask]]); y = np.concatenate([yt[mask], ym[mask]])
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                 class_weight="balanced", random_state=seed, n_jobs=4).fit(X, y)
    pri = np.array([prior[c] for c in clf.classes_], dtype=float)
    return clf, pri

def proba(clf, pri, X):
    Pp = clf.predict_proba(X)
    hi = Pp.max(axis=1) >= TAU
    Pc = Pp.copy(); Pc[~hi] = Pc[~hi] / (pri ** BETA)
    Pc = Pc / np.maximum(Pc.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    for j, c in enumerate(clf.classes_): out[:, CI[c]] = Pc[:, j]
    return out

def pick_flat(P): return [CLASSES[i] for i in P.argmax(axis=1)]
def pick_hier(P):
    G = P @ FMAT                          # 가족 합산
    gi = G.argmax(axis=1)
    out = []
    for r, g in zip(P, gi):
        mask = np.array([FI[f] == g for f in FAM])
        out.append(CLASSES[int(np.where(mask)[0][r[mask].argmax()])])
    return out

def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))

# 평가 피처 (미러 벡터는 저장돼 있지 않아 row 재로드 없이 불가 — X 를 그대로 두고
# 학습쪽 미러정의와 같은 변환이 필요하다. row_to_vec(mirror=True) 는 rows 가 필요하므로
# 여기서는 **모델 출력 확률의 클래스축 미러**로 등가 구현한다:
#   P_tta = (P(x) + mirror(P(x_mirror))) / 2  에서 x_mirror 예측 대신
#   학습을 미러 없이 두 번 하는 대신, 미러 학습 이중화가 이미 있으므로
#   P 의 클래스축 뒤집기 평균은 **항등이 아니고** 좌우 대칭 사전을 강제한다.
M2_NOTE = "M2 는 P 와 mirror(P) 의 평균 — 좌우 대칭 사전 강제 (피처 미러와 등가 아님, 근사)"

for tag, lab in (("trainoof", "292 OOF"), ("test", "test 83"), ("val", "val 42")):
    f = np.load(A / f"feat_{tag}.npz", allow_pickle=True)
    Xe, tr, cs = f["X"], list(f["truth"]), f["case"]
    res = collections.defaultdict(list)
    for sd in range(5):
        if tag == "trainoof":
            P = np.zeros((len(Xe), len(CLASSES)), dtype=np.float32)
            for fd in json.load(open(SP720)):
                va = set(fd["val"]); sel = np.array([c in va for c in cs])
                if not sel.any(): continue
                clf, pri = fit(np.array([c not in va for c in ct]), sd)
                P[sel] = proba(clf, pri, Xe[sel])
        else:
            clf, pri = fit(np.ones(len(Xt), bool), sd)
            P = proba(clf, pri, Xe)
        Ptta = 0.5 * (P + P[:, MIRROR_IDX])
        res["flat"].append(score(tr, pick_flat(P)))
        res["hier"].append(score(tr, pick_hier(P)))
        res["mtta"].append(score(tr, pick_flat(Ptta)))
        res["hier+mtta"].append(score(tr, pick_hier(Ptta)))
    print(f"\n[{lab}] 병변 {len(Xe)}")
    b = np.array(res["flat"]).mean(axis=0)
    print(f"  {'flat(현행)':<12} top1 {b[0]:.3f}  macroRec {b[1]:.3f}  MCC {b[2]:.4f}")
    for k in ("hier", "mtta", "hier+mtta"):
        m = np.array(res[k]).mean(axis=0)
        d = np.array(res[k])[:, 2] - np.array(res["flat"])[:, 2]
        print(f"  {k:<12} top1 {m[0]:.3f}  macroRec {m[1]:.3f}  MCC {m[2]:.4f}"
              f"   ΔMCC {m[2]-b[2]:+.4f}  {int((d>0).sum())}/5", flush=True)
print("\n" + M2_NOTE)
print("완료")
