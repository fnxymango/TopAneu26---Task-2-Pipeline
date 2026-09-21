"""C11 — 희귀 클래스 합성 (분기점에 가상 sac 배치).

c8/c9 진단: AChA(3.5) 정확도 0~20%, C7-nonBranch(3.6) 0%, C6-OA(3.2) 0%.
전부 3.4 Pcom-junction으로 빨려든다. 원인은 인코딩이 아니라 **샘플 수** —
AChA 4~5개 vs Pcom 29개. C9의 rel 블록(상대거리 인코딩) 실패가 이걸 증명했다.
공식지표는 52클래스 균등평균이라 이 희귀 클래스들이 흔한 클래스와 동일 가중치를 갖는다.

해법: c4가 뽑아둔 분기점 위치에 **가상 sac을 놓아 라벨된 학습샘플을 만든다.**
train 292케이스 전부에 R/L-AChA ostium 등이 존재하므로, 실제 병변이 4개뿐인 클래스도
수백 개의 합성 샘플을 얻는다. 병변 자체는 가짜지만 분류기가 배우는 것은
"sac이 이 해부학적 위치에 있으면 이 클래스"라는 관계이므로 유효하다.

**분기점 정의 클래스에만 적용한다.** 영역 클래스(3.3 C6-nonOA vs 3.6 C7-nonBranch 등)는
같은 혈관(ICA-C6-C7)에 매핑돼 위치가 모호해서 합성해도 서로 구분되지 않는다.

누수 방지: 합성 샘플은 **각 fold의 학습셋에만** 넣는다(평가 fold 케이스의 합성분은 제외).
평가는 언제나 실제 병변으로만 한다.

사용:
  python c11_synth_rare.py --feat <c10_feat_train.json> --bp-dir <c4출력> \
      --vessel-dir <참조혈관> [--per-class 40] [--folds 5]
"""
import argparse, json, collections
from pathlib import Path

import numpy as np
import nibabel as nib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8

# 위치클래스 -> 그 클래스를 정의하는 혈관 쌍(=c4 분기점 노드)
JUNCTION_CLASS = {
    "R-1.3 VA-PICA junction": {"R-VA", "R-PICA"},
    "L-1.3 VA-PICA junction": {"L-VA", "L-PICA"},
    "1.5 VA-BA junction": {"BA", "R-VA"},
    "R-1.7 BA-AICA junction": {"BA", "R-AICA"},
    "L-1.7 BA-AICA junction": {"BA", "L-AICA"},
    "R-1.9 BA-SCA junction": {"BA", "R-SCA"},
    "L-1.9 BA-SCA junction": {"BA", "L-SCA"},
    "1.10 BA tip": {"BA", "R-P1P2"},
    "R-3.2 ICA C6-OA-junction": {"R-ICA-C6-C7", "R-OA"},
    "L-3.2 ICA C6-OA-junction": {"L-ICA-C6-C7", "L-OA"},
    "R-3.4 ICA C7-Pcom-junction": {"R-ICA-C6-C7", "R-Pcom"},
    "L-3.4 ICA C7-Pcom-junction": {"L-ICA-C6-C7", "L-Pcom"},
    "R-3.5 ICA C7-AChA-junction": {"R-ICA-C6-C7", "R-AChA"},
    "L-3.5 ICA C7-AChA-junction": {"L-ICA-C6-C7", "L-AChA"},
    "R-3.7 ICA C7-terminus": {"R-ICA-C6-C7", "R-M1"},
    "L-3.7 ICA C7-terminus": {"L-ICA-C6-C7", "L-M1"},
    "4.1 Acom complex": {"Acom", "R-A1A2"},
    "R-5.3 M1-M2 junction": {"R-M1", "R-M2"},
    "L-5.3 M1-M2 junction": {"L-M1", "L-M2"},
}


def real_radius_dist(rows, spacing_guess=0.4):
    """실제 병변 voxel수 -> 등가 반지름(mm) 분포. 합성 sac 크기를 여기서 뽑는다."""
    v = np.array([r["n_vox"] for r in rows if r.get("n_vox")], dtype=float)
    vol = v * (spacing_guess ** 3)
    return np.clip((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0), 1.0, 8.0)


def synth_rows_for_case(cid, ves_dir, bp_dir, ves_names, radii, rng, per_class):
    """이 케이스의 분기점들에 가상 sac을 놓고 피처를 뽑는다."""
    bp_p = Path(bp_dir) / f"{cid}.json"
    ves_p = Path(ves_dir) / f"{cid}.nii.gz"
    if not bp_p.exists() or not ves_p.exists():
        return []
    nodes = json.load(open(bp_p))["nodes"]
    targets = []                       # (위치클래스, 중심mm)
    for name, pair in JUNCTION_CLASS.items():
        for nd in nodes:
            if pair <= set(nd["classes"]):
                targets.append((name, np.array(nd["centroid_mm"], dtype=float)))
                break
    if not targets:
        return []

    vi = nib.load(ves_p)
    ves = np.asanyarray(vi.dataobj)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    shape = ves.shape

    # 한 케이스의 모든 합성 병변을 하나의 마스크에 서로 다른 성분으로 그린다(추출 1회로 끝냄)
    mask = np.zeros(shape, dtype=np.uint8)
    order = []
    for name, cen in targets:
        r = float(rng.choice(radii))
        c_vox = np.round(cen / spacing).astype(int)
        rad_vox = np.ceil(r / spacing).astype(int)
        lo = np.maximum(c_vox - rad_vox, 0); hi = np.minimum(c_vox + rad_vox + 1, shape)
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d2 = (((zz - c_vox[0]) * spacing[0]) ** 2 + ((yy - c_vox[1]) * spacing[1]) ** 2
              + ((xx - c_vox[2]) * spacing[2]) ** 2)
        sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        put = (d2 <= r * r) & (sub == 0)          # 이미 그린 sac과 겹치지 않게
        if put.sum() < 3:
            continue
        sub[put] = 1
        order.append((name, cen))

    if not order:
        return []
    rows, _ = C5.extract_case_rows(mask, ves, spacing, ves_names, nodes, None)
    # 성분 라벨링 순서와 그린 순서가 다를 수 있으므로 중심 좌표로 매칭
    out = []
    for r in rows:
        if not r.get("pos"):
            continue
        # 이 성분의 중심에 가장 가까운 target 을 라벨로 삼는다
        # (extract_case_rows가 pos를 이미 계산해 뒀으므로 원 좌표는 bp_mm로 대신 판정)
        best, bd = None, 1e9
        cen_est = r.get("_cen")
        if cen_est is None:
            continue
        for name, cen in order:
            d = float(np.linalg.norm(np.array(cen_est) - cen))
            if d < bd:
                best, bd = name, d
        if best is not None and bd < 3.0:
            r["gt_loc"] = best
            r["case"] = cid
            r["synthetic"] = True
            out.append(r)
    return out[:per_class * len(JUNCTION_CLASS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--bp-dir", required=True)
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-cases", type=int, default=120,
                    help="합성에 쓸 train 케이스 수 (전량이면 느림)")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    C5.USE_POS = True
    rows = [r for r in json.load(open(args.feat)) if r.get("gt_loc")]
    ves_names = L.vessel_dense_names()
    ves_axis, _ = C5.build_feature_axes(ves_names)
    radii = real_radius_dist(rows)
    rng = np.random.default_rng(20260815)

    cache = Path(args.cache) if args.cache else Path(args.feat).parent / "c11_synth_rows.json"
    if cache.exists():
        synth = json.load(open(cache))
        print(f"[c11] 합성 샘플 캐시 로드 {len(synth)}개")
    else:
        train_ids = L.case_ids_by_split()[0][:args.max_cases]
        synth = []
        for i, cid in enumerate(train_ids, 1):
            synth.extend(synth_rows_for_case(cid, args.vessel_dir, args.bp_dir,
                                             ves_names, radii, rng, 4))
            if i % 20 == 0 or i == len(train_ids):
                print(f"  {i}/{len(train_ids)}  누적 합성 {len(synth)}", flush=True)
        json.dump(synth, open(cache, "w"), ensure_ascii=False)
        print(f"[c11] 합성 {len(synth)}개 -> {cache}")

    cnt = collections.Counter(r["gt_loc"] for r in synth)
    print("  합성 클래스 분포 상위:", dict(cnt.most_common(6)))

    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    y = np.array([r["gt_loc"] for r in rows])

    def vec(r, mirror=False):
        return C5.row_to_vec(r, ves_axis, mirror=mirror)

    print(f"\n{'실험':<28}{'top-1':>8}{'macroRec':>10}{'ICA':>8}")
    for w in (0, 1, 2):          # 합성 샘플 반복 횟수 (0 = 미사용 기준선)
        pred = np.empty(len(rows), dtype=object)
        for tr, te in GroupKFold(n_splits=args.folds).split(np.zeros(len(rows)), y, groups):
            tr_cases = {rows[i]["case"] for i in tr}
            X, Y = [], []
            for i in tr:
                X.append(vec(rows[i])); Y.append(y[i])
                X.append(vec(rows[i], True)); Y.append(C5.mirror_name(y[i]))
            for _ in range(w):
                for s in synth:
                    if s["case"] in tr_cases:      # 평가 fold 케이스는 제외 (누수 방지)
                        X.append(vec(s)); Y.append(s["gt_loc"])
            X = np.array(X); Y = np.array(Y)
            clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                         random_state=0, n_jobs=-1).fit(X, Y)
            prior = collections.Counter(Y)
            pri = np.array([prior[c] for c in clf.classes_], dtype=float)
            Xte = np.array([vec(rows[i]) for i in te])
            P = clf.predict_proba(Xte) / pri      # beta=1.0 (C10 최적)
            pred[te] = clf.classes_[np.argmax(P, axis=1)]
        t1 = float(np.mean(y == pred))
        mr, _ = C8.macro_recall(y, pred)
        ica = [(a, b) for a, b in zip(y, pred) if C8.group_of(a) == "3"]
        ia = float(np.mean([a == b for a, b in ica]))
        print(f"{'합성 x' + str(w):<28}{t1:>8.3f}{mr:>10.3f}{ia:>8.3f}")

    print("[c11] 완료")


if __name__ == "__main__":
    main()
