#!/usr/bin/env python3
# ── V4-I (2026-09-14) · deep_ica.py 를 **추론 조건**으로 옮긴 복사본 ─────────────────────────────
# 바뀐 것 3개뿐: 축 기하 = 예측 혈관 vespp_train · 접촉거리·RF = 예측혈관 학습표 c10_feat_train_predves_NEW · 캐시 이름.
# 왜: GT 조건 스크리닝에서 기하 룰 τ2.0 이 RF 대비 +6.9%p 였지만 거리 임계 룰은 V1-B 에서 추론 이동(0.2→0.72mm)에 죽었다.
#     추론과 같은 혈관에서 재도 살아남는지를 본다. 절단 0.493/0.814 와 τ 는 GT 스크리닝 값 그대로(재튜닝 금지).
# 판정규칙(결과 보기 전 고정): 주 = "M1g 기하 룰 τ2.0 (1등 ICA원위일 때)" 정확도 ≥ 같은 조건 M0 + 3.0%p.
#   통과 시에만 c5 룰 구현 → e2e 5시드(사용자 확인 후). 미달이면 ICA 룰 축을 닫는다.
"""ICA 원위(3.2~3.7) 심층 — V2-A 가 실패한 뒤, **같은 정보를 다른 방식으로 쓰면** 살아나는가 (train CV 스크리닝).

V2-A 는 C6/C7/terminus 구획 정보를 **124차원 전체 RF 의 12차원**으로 줬다. 52클래스 전체를
한 번에 배우는 RF 안에서 ICA 병변 ~100개로 그 12차원을 배우기엔 신호가 묽다. 반면 V2-1 은
"병변이 붙은 구획"만으로 병변 클래스가 84~92% 결정된다고 했다. 둘 사이의 간극이 **방법** 문제인지 잰다.

방법 (모두 train 271행 · 5겹 CV × 5시드 OOF · c5 와 같은 RF/미러/β·τ 설정)
  M0  RF 1등 그대로                          (기준)
  M1  순수 기하 룰   구획(lesion_t, 절단 0.493/0.814 = V2-2 train 추정) → 구획 안에서
                     OA 접촉 → 3.2 / 없으면 3.3 · Pcom 접촉 → 3.4 / AChA 접촉 → 3.5 / 없으면 3.6 · terminus → 3.7
  M2  **거부(veto)** RF 순위에서 **기하 구획과 모순되지 않는** 첫 클래스를 고른다 (RF 1등이 ICA 원위일 때만)
  M3  **ICA 전용 소형 RF** ICA 원위 행만으로, 스칼라 피처 ~14개(lesion_t · 분지 t · 분지 접촉거리 …)로 학습
                     → RF 1등이 ICA 원위일 때만 그 결정을 대체
  쪽(R/L)은 라벨이 아니라 **병변이 더 가까운 ICA 축**으로 정한다(추론 때와 같은 조건).
⚠ CV 는 스크리닝이다(PROJECT_RULES.md 2장). 채택은 e2e 5시드 test·val 로만 한다.
"""
import json, os, sys, re, collections
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
sys.path.insert(0, D); sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts")
os.environ.setdefault("TOPANEU_ROOT", R)
import ica_split_lib as IL

A = f"{R}/code/sblee/nnunet/analysis"
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
LOC = {int(k): v for k, v in S["location_classes"].items()}
NAME2ID = {v: k for k, v in LOC.items()}
SEG = {"3.2": 0, "3.3": 0, "3.4": 1, "3.5": 1, "3.6": 1, "3.7": 2}
C1, C2 = 0.493, 0.814


def key(name):
    m = re.match(r"^(?:[RL]-)?(\d+\.\d+)\s", name or "")
    return m.group(1) if m else None


# ── 1. 기하 측정 (GT 혈관 · train ICA 원위 병변) ──────────────────────────────
def geo_case(cid, rows):
    im = nib.load(f"{R}/experiments/_c1_realpred/vespp_train/{cid}.nii.gz")
    ves = np.asanyarray(im.dataobj); sp = np.array(im.header.get_zooms()[:3], float)
    loc = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz").dataobj)
    axes = {s: IL.side_axis(ves, sp, s) for s in ("R", "L")}
    out = {}
    for r in rows:
        k = NAME2ID.get(r["gt_loc"])
        lab, n = ndimage.label(loc == k, structure=np.ones((3, 3, 3)))
        cen = np.array(r["_cen"])
        best = None
        for i in range(1, n + 1):
            c = np.argwhere(lab == i).mean(0) * sp
            dd = float(np.linalg.norm(c - cen))
            if best is None or dd < best[0]:
                best = (dd, i)
        if best is None or best[0] > 3.0:
            continue
        les = lab == best[1]
        per = {}
        for s, ax in axes.items():
            if ax is None:
                continue
            lc = les[ax["sl"]]
            if not lc.any():
                continue
            d = ndimage.distance_transform_edt(~lc, sampling=sp)
            dmin = float(np.nanmin(np.where(ax["body"], d, np.inf)))
            if dmin > 5.0:
                continue
            near = ax["body"] & (d <= dmin + 1.0)
            per[s] = dict(dmin=dmin, t=float(np.nanmedian(ax["t"][near])), bt=ax["bt"], len=ax["len_mm"])
        if per:
            s = min(per, key=lambda z: per[z]["dmin"])
            out[(r["case"], r["lesion_mask_idx"])] = dict(side=s, **per[s])
    return out


def geo_all(rows):
    cache = f"{D}/deep_ica_pv_geo.json"
    if os.path.exists(cache):
        return {tuple(json.loads(k)): v for k, v in json.load(open(cache)).items()}
    by = collections.defaultdict(list)
    for r in rows:
        if key(r["gt_loc"]) in SEG:
            by[r["case"]].append(r)
    import multiprocessing as mp
    res = {}
    with mp.Pool(6) as pool:
        for o in pool.starmap(geo_case, list(by.items())):
            res.update(o)
    json.dump({json.dumps(list(k)): v for k, v in res.items()}, open(cache, "w"))
    return res


# ── 2. OOF 순위 ─────────────────────────────────────────────────────────────
def oof(rows):
    cache = f"{D}/deep_ica_pv_oof.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    import c5_location_v2 as C
    from sklearn.model_selection import KFold
    C.USE_POS = True; C.CONF_TAU = 0.5; C.CONF_BETA_HI = 0.0
    ves_axis = [C.L.vessel_dense_names()[i] for i in sorted(C.L.vessel_dense_names())]
    out = {}
    for sd in range(5):
        os.environ["CLF_SEED"] = str(sd)
        for tr, te in KFold(5, shuffle=True, random_state=sd).split(rows):
            m = C.fit_model([rows[i] for i in tr], ves_axis, kind="rf", mirror=True)
            for i in te:
                nm, p, _ = C.predict_ranked(m, rows[i], 0.5)
                out.setdefault(str(i), []).append(dict(seed=sd, names=nm[:10] if nm else [],
                                                       p=[float(x) for x in p[:10]] if nm else []))
        print(f"  OOF 시드 {sd} 완료", flush=True)
    json.dump(out, open(cache, "w"))
    return out


# ── 3. 방법들 ───────────────────────────────────────────────────────────────
def geo_segment(g):
    return 0 if g["t"] < C1 else (1 if g["t"] < C2 else 2)


def contact(r, side, vessel, tau):
    v = (r.get("dist_mm") or {}).get(f"{side}-{vessel}")
    return v is not None and v <= tau


def rule_class(r, g, tau):
    s, seg = g["side"], geo_segment(g)
    if seg == 0:
        k = "3.2" if contact(r, s, "OA", tau) else "3.3"
    elif seg == 1:
        k = "3.4" if contact(r, s, "Pcom", tau) else ("3.5" if contact(r, s, "AChA", tau) else "3.6")
    else:
        k = "3.7"
    return next(v for v in LOC.values() if v.startswith(f"{s}-{k} "))


def veto(names, g):
    seg = geo_segment(g)
    for nm in names:
        k = key(nm)
        if k in SEG and SEG[k] == seg and nm[0] == g["side"]:
            return nm
    return names[0] if names else None


FEATS = ["t", "len", "tOA", "tPcom", "tAChA", "tA1", "tM1", "t-Pcom", "t-A1", "dOA", "dPcom", "dAChA", "dM1", "dA1"]


def feat(r, g):
    bt = g["bt"]; s = g["side"]; dm = r.get("dist_mm") or {}
    f = lambda v: -1.0 if v is None else float(v)
    dd = lambda n: f(dm.get(f"{s}-{n}", 15.0))
    return [g["t"], g["len"], f(bt.get("OA")), f(bt.get("Pcom")), f(bt.get("AChA")), f(bt.get("A1A2")),
            f(bt.get("M1")), (g["t"] - bt["Pcom"]) if bt.get("Pcom") is not None else 0.0,
            (g["t"] - bt["A1A2"]) if bt.get("A1A2") is not None else 0.0,
            dd("OA"), dd("Pcom"), dd("AChA"), dd("M1"), dd("A1A2")]


def main():
    rows = json.load(open(f"{A}/c10_feat_train_predves_NEW.json"))
    rows = [r for r in rows if r.get("gt_loc")]
    geo = geo_all(rows)
    ranks = oof(rows)
    idx = [i for i, r in enumerate(rows) if key(r["gt_loc"]) in SEG]
    have = [i for i in idx if (rows[i]["case"], rows[i]["lesion_mask_idx"]) in geo]
    print(f"# ICA 원위 심층 — train CV 스크리닝\n\nICA 원위 학습 병변 {len(idx)} · 기하 측정 성공 {len(have)}\n")
    gs = {i: geo[(rows[i]["case"], rows[i]["lesion_mask_idx"])] for i in have}
    side_ok = sum(gs[i]["side"] == rows[i]["gt_loc"][0] for i in have)
    seg_ok = sum(geo_segment(gs[i]) == SEG[key(rows[i]["gt_loc"])] for i in have)
    print(f"- 쪽 판정(가까운 축) 정답 {side_ok}/{len(have)} · 구획 판정 정답 {seg_ok}/{len(have)} = {seg_ok/len(have):.0%}\n")

    # M3 준비: ICA 원위 행만으로 소형 RF — 같은 OOF 폴드 규약(시드별 KFold)으로 평가
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold
    X = np.array([feat(rows[i], gs[i]) for i in have]); Y = np.array([key(rows[i]["gt_loc"]) for i in have])
    spec = collections.defaultdict(list)       # row -> [pred key per seed]
    for sd in range(5):
        for tr, te in KFold(5, shuffle=True, random_state=sd).split(X):
            clf = RandomForestClassifier(300, class_weight="balanced", random_state=sd, min_samples_leaf=1)
            clf.fit(X[tr], Y[tr])
            for j, pk in zip(te, clf.predict(X[te])):
                spec[have[j]].append(pk)
    imp = RandomForestClassifier(300, class_weight="balanced", random_state=0).fit(X, Y).feature_importances_

    res = collections.defaultdict(lambda: collections.Counter())
    percls = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for i in have:
        r, g = rows[i], gs[i]
        gt = r["gt_loc"]; k = key(gt)
        for sd, rk in enumerate(ranks[str(i)]):
            names = rk["names"]
            if not names:
                continue
            top1 = names[0]
            guard = key(top1) in SEG
            cand = {"M0 RF 1등": top1,
                    "M1 기하 룰 τ0.5": rule_class(r, g, 0.5),
                    "M1 기하 룰 τ1.0": rule_class(r, g, 1.0),
                    "M1 기하 룰 τ2.0": rule_class(r, g, 2.0),
                    "M1g 기하 룰 τ2.0 (1등 ICA원위일 때)": rule_class(r, g, 2.0) if guard else top1,
                    "M2 veto (1등 ICA원위일 때)": veto(names, g) if guard else top1,
                    "M3 ICA 전용 RF (1등 ICA원위일 때)": (f"{g['side']}-{spec[i][sd]}" if guard else top1)}
            cand["M3 ICA 전용 RF (1등 ICA원위일 때)"] = next(
                (v for v in LOC.values() if v.startswith(cand["M3 ICA 전용 RF (1등 ICA원위일 때)"] + " ")),
                cand["M3 ICA 전용 RF (1등 ICA원위일 때)"]) if guard else top1
            for m, nm in cand.items():
                ok = nm == gt
                res[m]["ok"] += ok; res[m]["n"] += 1
                percls[m][k][0] += ok; percls[m][k][1] += 1

    print("## 1. 방법별 정확도 (ICA 원위 train 병변 × 5시드 OOF)\n")
    print("| 방법 | 정답 | 정확도 |\n|---|---|---|")
    for m, c in res.items():
        print(f"| {m} | {c['ok']}/{c['n']} | **{c['ok']/max(c['n'],1):.1%}** |")
    print("\n## 2. 클래스별\n")
    ms = list(res)
    print("| 클래스 | " + " | ".join(ms) + " |\n|" + "---|" * (len(ms) + 1))
    for k in sorted(SEG):
        print(f"| {k} | " + " | ".join(f"{percls[m][k][0]}/{percls[m][k][1]}" for m in ms) + " |")
    print("\n## 3. ICA 전용 RF 피처 중요도\n")
    for f_, v in sorted(zip(FEATS, imp), key=lambda x: -x[1]):
        print(f"- {f_} {v:.3f}")

    # 비ICA 병변에 대한 부작용: RF 1등이 ICA 원위인데 GT 가 ICA 원위가 아닌 병변 수
    fp = sum(1 for i, r in enumerate(rows) if key(r["gt_loc"]) not in SEG
             for rk in ranks[str(i)] if rk["names"] and key(rk["names"][0]) in SEG)
    print(f"\n참고 · GT 가 ICA 원위가 아닌데 RF 1등이 ICA 원위인 OOF 판정 {fp}회 — M2/M3 가 여기서는 원래도 오답이라 손해가 없다.")


if __name__ == "__main__":
    main()
