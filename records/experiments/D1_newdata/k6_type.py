#!/usr/bin/env python3
"""K6-1 — 동맥류 유형(낭형 vs 비낭형=방추형·박리형)을 형상만으로 예측할 수 있나 (train 관문 · test·val 안 봄).

근거: train 비낭형 41개 중 34개가 후순환(1.1 VA trunk 17/18 · 1.4 BA trunk 9/11 · 1.5 4/6). 후순환 정확도 최하(46%).
추론 때 유형 라벨은 없으므로, 분류기에 쓰려면 먼저 형상에서 예측돼야 한다.

형상 피처(학습표와 같은 GT 병변 · GT 혈관 · 낭 복셀은 혈관에서 뺀다 — V5 에서 GT 혈관이 낭을 덮는 것 확인):
  log 부피 · 주축 길이비(λ1/λ2 · λ2/λ3) · 원래 GT 혈관 라벨이 덮은 병변 비율 · 병변 주축과 모혈관 축의 |cos| ·
  모혈관 반경 · 등가지름/모혈관 반경 · 목 복셀 비율(혈관 인접 병변 복셀 / 병변) · V5 F1 의 돌출방향–모혈관 |cos|
관문(고정): 케이스 단위 5겹 × 시드 5 OOF AUC(비낭형 양성) 평균 ≥ 0.85 → K6-2(분류기 블록) 로. 미달이면 K6 중단.
출력: k6_type_feat.json · k6_type_gate.json
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
A = f"{R}/code/sblee/nnunet/analysis"
ST = np.ones((3, 3, 3), bool)


def case(cid, rows):
    li = nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz")
    loc = np.asanyarray(li.dataobj); sp = np.array(li.header.get_zooms()[:3], float)
    ves = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz").dataobj)
    ty = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/type_masks/{cid}.nii.gz").dataobj)
    lab, _ = ndimage.label(loc > 0, structure=ST)
    v5 = json.load(open(f"{D}/v5_feat_train.json"))
    out = {}
    for r in rows:
        k = f"{cid}|{r['lesion_mask_idx']}"
        m = lab == r["lesion_mask_idx"]
        idx = np.argwhere(m)
        t = ty[m]; t = t[t > 0]
        label = int(np.bincount(t).argmax()) if t.size else 0
        pts = idx * sp; nv = len(idx)
        if nv >= 4:
            w = np.sort(np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)))[::-1]
            w = np.maximum(w, 1e-6)
            e12, e23 = float(np.sqrt(w[0] / w[1])), float(np.sqrt(w[1] / w[2]))
            major = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, -1]
        else:
            e12 = e23 = 1.0; major = None
        in_ves = float((ves[m] > 0).mean())
        lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 13, loc.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        mc = m[sl]; vc = np.where(mc, 0, ves[sl]); vb = vc > 0
        neck_frac = float((mc & ndimage.binary_dilation(vb, structure=ST)).sum() / max(mc.sum(), 1))
        parent = v5[k]["parent"]; rp = v5[k]["f2"][0]
        cosm = 0.0
        if major is not None and parent:
            pc = (np.argwhere(vc == parent) + lo) * sp
            cen = pts.mean(0)
            sel = pc[np.linalg.norm(pc - cen, axis=1) <= 8.0]
            if len(sel) >= 20:
                pa = np.linalg.eigh(np.cov((sel - sel.mean(0)).T))[1][:, -1]
                cosm = abs(float(np.dot(pa, major)))
        vol = nv * float(np.prod(sp)); deq = 2 * (3 * vol / (4 * np.pi)) ** (1 / 3)
        feat = [np.log10(max(vol, 1e-3)), e12, e23, in_ves, cosm, rp, deq / rp if rp > 0 else -1.0, neck_frac, v5[k]["f1"][3]]
        out[k] = dict(x=[float(v) for v in feat], type=label)
    return out


def main():
    import multiprocessing as mp
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    cache = f"{D}/k6_type_feat.json"
    if os.path.exists(cache):
        F = json.load(open(cache))
    else:
        F = {}
        with mp.Pool(8) as p:
            for o in p.starmap(case, list(by.items()), chunksize=1):
                F.update(o)
        json.dump(F, open(cache, "w"))
    keys = [f"{r['case']}|{r['lesion_mask_idx']}" for r in rows]
    X = np.array([F[k]["x"] for k in keys]); y = np.array([int(F[k]["type"] in (2, 3)) for k in keys])
    cs = np.array([r["case"] for r in rows]); ucs = sorted(set(cs))
    aucs = []
    for sd in range(5):
        perm = np.random.default_rng(sd).permutation(ucs); fold = {c: i % 5 for i, c in enumerate(perm)}
        f = np.array([fold[c] for c in cs]); pr = np.zeros(len(y))
        for k in range(5):
            tr, te = f != k, f == k
            clf = RandomForestClassifier(500, class_weight="balanced", random_state=sd, n_jobs=8).fit(X[tr], y[tr])
            pr[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, pr))
    names = ["log부피", "주축비12", "주축비23", "GT혈관이 덮은 비율", "주축-모혈관|cos|", "모혈관반경", "등가지름/모혈관반경", "목비율", "돌출-모혈관|cos|"]
    imp = RandomForestClassifier(500, class_weight="balanced", random_state=0, n_jobs=8).fit(X, y).feature_importances_
    ok = float(np.mean(aucs)) >= 0.85
    print(f"# K6-1 유형 예측 관문 — train {len(y)}병변 · 비낭형 {int(y.sum())}\n")
    print(f"OOF AUC (케이스 5겹 × 시드5): {' '.join(f'{a:.3f}' for a in aucs)} → 평균 **{np.mean(aucs):.3f}**\n")
    print("| 피처 | 중요도 | 낭형 중앙 | 비낭형 중앙 |\n|---|---|---|---|")
    for j in np.argsort(-imp):
        print(f"| {names[j]} | {imp[j]:.3f} | {np.median(X[y == 0, j]):.2f} | {np.median(X[y == 1, j]):.2f} |")
    print(f"\n**관문(AUC ≥ 0.85) → {'통과 · K6-2 진행' if ok else '미달 · K6 중단'}**")
    json.dump(dict(aucs=aucs, mean=float(np.mean(aucs)), ok=ok), open(f"{D}/k6_type_gate.json", "w"))


if __name__ == "__main__":
    main()
