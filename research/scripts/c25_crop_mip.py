"""C25 — 크롭 분류기를 3D CNN에서 MIP+RF 로 교체 (2026-08-17).

C24 에서 두 숫자가 모순됐다:
  게이트2 (MIP 최대투영 + RF, 학습이랄 것도 없음)  영상채널만 top-1 **0.410**
  본선    (3D CNN, 120 epoch)                       top-1 **0.134**
268샘플로 96^3 3D CNN 을 돌린 게 원인이다. **영상에 정보는 있고 CNN 이 못 꺼낸 것**이므로
이미 0.410 이 측정된 경로를 본선으로 올린다. 기하 피처 112차원 전체가 같은 CV 에서 0.496 이니
원본 화소만으로 거의 맞먹는다.

미러 증강은 쓰지 않는다 — 크롭 축 중 어느 것이 좌우인지 케이스마다 보장되지 않아
잘못 뒤집으면 R-/L- 라벨이 조용히 오염된다. 게이트2 의 0.410 도 미러 없이 나온 수치다.

산출:
  analysis/c25_report.json           CV 결과 + 최적 가중치 w
  analysis/c25_prob_<split>.json     {case__lesionidx: {클래스: 확률}}  — c5 eval --crop-prob 로 투입

사용: python c25_crop_mip.py --crop-dir <train크롭> --feat <c10_feat_train.json> \
        --apply val_gt=<dir> --apply test_det=<dir> ...
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8
import c24_train_eval as C24

POOL = 24        # MIP 을 24x24 로 축소. 96x96 원본은 268샘플 대비 차원이 과하다.


def load_crops(crop_dir, need_label=True):
    meta = json.load(open(Path(crop_dir) / "meta.json"))
    if need_label:
        meta = [m for m in meta if m.get("gt_loc")]
    if not meta:
        return [], np.zeros((0, 3, 1, 1, 1), np.float32)
    X = np.stack([np.load(Path(crop_dir) / f"{m['key']}.npz")["x"] for m in meta])
    return meta, X.astype(np.float32)


def mipfeat(X, channels):
    """(N,C,D,H,W) -> 3면 최대투영을 POOL 해상도로 줄여 평탄화 + 채널별 요약통계."""
    F = C24.mip3(X[:, channels])                       # (N, C*3, H, W)
    F = ndi.zoom(F, [1, 1, POOL / F.shape[2], POOL / F.shape[3]], order=1)
    V = X[:, channels].reshape(len(X), len(channels), -1)
    stats = np.concatenate([V.mean(2), V.std(2),
                            np.percentile(V, 90, axis=2),
                            np.percentile(V, 99, axis=2)], axis=1)
    return np.concatenate([F.reshape(len(X), -1), stats], axis=1)


def oof_probs(F, y, groups, folds=5):
    """환자단위 CV 의 out-of-fold 확률. 반환 (클래스배열, 확률행렬)."""
    classes = sorted(set(y))
    ci = {c: i for i, c in enumerate(classes)}
    P = np.zeros((len(y), len(classes)), np.float32)
    for tr, te in GroupKFold(n_splits=folds).split(F, y, groups):
        clf = RandomForestClassifier(n_estimators=600, class_weight="balanced",
                                     min_samples_leaf=1, random_state=0, n_jobs=-1)
        clf.fit(F[tr], y[tr])
        pr = clf.predict_proba(F[te])
        for j, c in enumerate(clf.classes_):
            P[te, ci[c]] = pr[:, j]
    return np.array(classes), P


def score(y, pred):
    t1 = float(np.mean(y == pred))
    mr, _ = C8.macro_recall(y, pred)
    ica = [(a, b) for a, b in zip(y, pred) if C8.group_of(a) == "3"]
    return t1, mr, (float(np.mean([a == b for a, b in ica])) if ica else 0.0)


def align(classes_src, P_src, classes_dst):
    """확률행렬을 목표 클래스 축으로 재배치."""
    idx = {c: j for j, c in enumerate(classes_src)}
    out = np.zeros((len(P_src), len(classes_dst)), np.float32)
    for k, c in enumerate(classes_dst):
        if c in idx:
            out[:, k] = P_src[:, idx[c]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-dir", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--apply", action="append", default=[],
                    help="name=dir 형식. 학습된 크롭 RF 로 확률을 뽑아 json 으로 내보낸다.")
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()

    A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
    meta, X = load_crops(a.crop_dir)
    y = np.array([m["gt_loc"] for m in meta])
    pm = C8.patient_map()
    groups = np.array([pm.get(m["case"], m["case"]) for m in meta])
    keys = [f"{m['case']}__{m['lesion_idx']}" for m in meta]
    print(f"[c25] 크롭 {len(meta)} · 크기 {X.shape[2:]} · 클래스 {len(set(y))}\n")

    # ── 크롭 분기 후보 ─────────────────────────────────────────────────
    feats = {"영상만 (ch0)": [0], "영상+혈관 (ch0,1,2)": [0, 1, 2]}
    crop_res = {}
    print(f"{'크롭 피처':<24}{'차원':>7}{'top-1':>9}{'macroRec':>10}{'ICA':>8}")
    for name, ch in feats.items():
        F = mipfeat(X, ch)
        cls, P = oof_probs(F, y, groups)
        t1, mr, ic = score(y, cls[np.argmax(P, 1)])
        print(f"{name:<24}{F.shape[1]:>7}{t1:>9.3f}{mr:>10.3f}{ic:>8.3f}")
        crop_res[name] = {"channels": ch, "dim": int(F.shape[1]),
                          "top1": t1, "macro_recall": mr, "ica": ic,
                          "_cls": cls, "_P": P}
    best_crop = max(crop_res, key=lambda k: crop_res[k]["macro_recall"])
    print(f"  -> 크롭 분기 채택: {best_crop}\n")

    # ── 기하 RF (C24 와 동일 경로) ─────────────────────────────────────
    classes_g, P_geo_all, sel = C24.geo_probs(a.feat, keys)
    classes_g = np.asarray(classes_g)          # geo_probs 는 리스트를 준다 (argmax 인덱싱용)
    ok = [i for i, s in enumerate(sel) if s is not None]
    if len(ok) < len(meta) * 0.8:
        raise SystemExit(f"기하 피처 매칭 {len(ok)}/{len(meta)} — 중단")
    P_geo = P_geo_all[[sel[i] for i in ok]]
    yo = y[ok]
    prior = collections.Counter(yo)
    pri = np.array([max(prior.get(c, 1), 1) for c in classes_g], float)

    # ── 앙상블 스윕 ────────────────────────────────────────────────────
    Pc = align(crop_res[best_crop]["_cls"], crop_res[best_crop]["_P"][ok], classes_g)
    print(f"{'w(크롭)':>8}{'top-1':>9}{'macroRec':>10}{'ICA':>8}")
    sweep, best = [], None
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        P = w * Pc + (1 - w) * P_geo
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
        if a.beta > 0:
            P = P / (pri ** a.beta)
        t1, mr, ic = score(yo, classes_g[np.argmax(P, 1)])
        print(f"{w:>8.1f}{t1:>9.3f}{mr:>10.3f}{ic:>8.3f}")
        row = {"w": w, "top1": t1, "macro_recall": mr, "ica": ic}
        sweep.append(row)
        if best is None or mr > best["macro_recall"]:
            best = row
    base = sweep[0]["macro_recall"]
    print(f"\n[최적] w={best['w']} · macro-recall {best['macro_recall']:.3f} "
          f"(기하 단독 {base:.3f}, {best['macro_recall'] / base - 1:+.1%}) "
          f"· top-1 {best['top1']:.3f} (기하 단독 {sweep[0]['top1']:.3f})")

    # ── 최종 학습 후 val/test 크롭에 적용 ──────────────────────────────
    ch = crop_res[best_crop]["channels"]
    clf = RandomForestClassifier(n_estimators=600, class_weight="balanced",
                                 random_state=0, n_jobs=-1).fit(mipfeat(X, ch), y)
    exported = {}
    for spec in a.apply:
        name, d = spec.split("=", 1)
        m2, X2 = load_crops(d, need_label=False)
        if not m2:
            print(f"[적용] {name}: 크롭 없음 — 건너뜀"); continue
        P2 = clf.predict_proba(mipfeat(X2, ch))
        out = {f"{m['case']}__{m['lesion_idx']}":
               {str(c): round(float(p), 5) for c, p in zip(clf.classes_, row) if p > 1e-4}
               for m, row in zip(m2, P2)}
        p = A / f"c25_prob_{name}.json"
        json.dump(out, open(p, "w"))
        exported[name] = str(p)
        print(f"[적용] {name}: {len(out)}건 -> {p}")

    rep = {"n_crops": len(meta), "pool": POOL,
           "crop_variants": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                             for k, v in crop_res.items()},
           "best_crop_feature": best_crop, "beta": a.beta,
           "sweep": sweep, "best": best, "geo_only": sweep[0], "exported": exported}
    json.dump(rep, open(A / "c25_report.json", "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {A / 'c25_report.json'}")


if __name__ == "__main__":
    main()
