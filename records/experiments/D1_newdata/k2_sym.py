#!/usr/bin/env python3
"""K2 — 좌우 대칭 앵커 보간 · train OOF 스크리닝 (test·val 안 봄).

문제: ICA 곁가지 분기점(ICA-C6-C7 ↔ OA · Pcom · AChA)이 그래프에서 자주 빠진다(참조 마스크에서도 41~60%, C41).
빠진 쪽 병변은 3.2/3.4/3.5 를 가를 분기점 거리가 비어 3.3/3.6 과 입력이 같아진다.
보간: 한쪽 분기점만 있으면, C10 랜드마크 좌표계의 정중면(원점 O · x=R→L)에 대해 거울상으로 옮긴 뒤
     빠진 쪽 ICA-C6-C7 혈관 복셀 중 가장 가까운 점에 붙인다(6mm 넘으면 버림). 노드에 "sym": true.
pseudo 앵커(C60 §38 · 끊긴 곁가지 기시부)와 다르다 — 곁가지 자체가 안 그려진 경우를 겨냥. C60 §30 "미시도".
학습표와 같은 방식: 참조 그래프(all_ref) + GT 혈관 에 적용하고, GT 병변 좌표로 bp_mm 을 다시 잰다(기존보다 가까울 때만 갱신).

── 관문 (결과 보기 전 고정 · 2026-09-15 · V5 와 같은 틀) ─────────────────────
 팔: base(기준 bp_mm) · sym(보간 bp_mm) · 케이스 5겹 × 시드 5 · c5 fit_model/predict_one 그대로
 통과 = 평균 Δmacro-recall ≥ +0.02 ∧ 평균 Δtop1 ≥ −0.005 ∧ Δmacro-recall>0 시드 ≥ 4/5 → e2e(추론 그래프 vespp_* 에도 같은 보간)
 표적 구간(3.2~3.6) top1 은 참고.
출력: k2_sym_train.json (행별 새 bp_mm) · k2_screen.out
"""
import json, os, sys, collections, re
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D)
os.environ.setdefault("TOPANEU_ROOT", R)
ST = np.ones((3, 3, 3), bool)
BRANCH = ("OA", "Pcom", "AChA")
SNAP_MM = 6.0


def augment(nodes, ves, sp, nm2id):
    import c5_location_v2 as C5
    fr = C5.landmark_frame(nodes)
    if fr is None:
        return [], {}
    O, Ax, s, lm = fr
    xh = Ax[0]
    have = {}
    for nd in nodes:
        cs = set(nd["classes"])
        for sd in ("R", "L"):
            for b in BRANCH:
                if {f"{sd}-ICA-C6-C7", f"{sd}-{b}"} <= cs and (sd, b) not in have and nd.get("valid", True):
                    have[(sd, b)] = np.array(nd["centroid_mm"], float)
    added, stat = [], collections.Counter()
    ica_pts = {}
    for b in BRANCH:
        for sd, od in (("R", "L"), ("L", "R")):
            if (sd, b) in have or (od, b) not in have:
                continue
            stat[f"결측{b}"] += 1
            if sd not in ica_pts:
                vid = nm2id.get(f"{sd}-ICA-C6-C7")
                pts = np.argwhere(ves == vid) * sp if vid is not None else np.zeros((0, 3))
                ica_pts[sd] = pts
            pts = ica_pts[sd]
            if len(pts) == 0:
                continue
            p = have[(od, b)]
            q = p - 2.0 * np.dot(p - O, xh) * xh
            dd = np.linalg.norm(pts - q, axis=1); j = int(dd.argmin())
            if dd[j] > SNAP_MM:
                stat[f"붙임실패{b}"] += 1
                continue
            added.append({"classes": [f"{sd}-ICA-C6-C7", f"{sd}-{b}"], "centroid_mm": pts[j].tolist(),
                          "valid": True, "sym": True, "snap_mm": float(dd[j])})
            stat[f"보간{b}"] += 1
    return added, stat


def case_job(cid, rows):
    import c5_location_v2 as C5, d9xx_lib as L
    nm2id = {n: i for i, n in L.vessel_dense_names().items()}
    li = nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz")
    loc = np.asanyarray(li.dataobj); sp = np.array(li.header.get_zooms()[:3], float)
    ves = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz").dataobj)
    nodes = C5.load_bp(f"{R}/experiments/_c4_bpgraph/all_ref", cid)
    added, stat = augment(nodes, ves, sp, nm2id)
    lab, _ = ndimage.label(loc > 0, structure=ST)
    out = {}
    for r in rows:
        bp = [np.inf if v is None else v for v in r["bp_mm"]]
        if added:
            coords = np.argwhere(lab == r["lesion_mask_idx"]) * sp
            new = C5.lesion_bp_features(coords, added)
            bp = [min(a, b) for a, b in zip(bp, new)]
        out[f"{cid}|{r['lesion_mask_idx']}"] = [None if not np.isfinite(v) else round(float(v), 3) for v in bp]
    return out, stat


def run_arm(arm):
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ves_axis = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    if arm == "sym":
        new = json.load(open(f"{D}/k2_sym_train.json"))
        for r in rows:
            r["bp_mm"] = new[f"{r['case']}|{r['lesion_mask_idx']}"]
    cases = sorted({r["case"] for r in rows})
    out = {}
    for sd in range(5):
        os.environ["CLF_SEED"] = str(sd)
        perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
        pred = {}
        for k in range(5):
            m = C5.fit_model([r for r in rows if fold[r["case"]] != k], ves_axis, kind="rf", mirror=True)
            for i, r in enumerate(rows):
                if fold[r["case"]] == k:
                    pred[i] = C5.predict_one(m, r, 0.5)
        out[sd] = [pred[i] for i in range(len(rows))]
    return arm, out, [r["gt_loc"] for r in rows]


def code(nm):
    b = re.sub(r"^[RL]-", "", nm); c = b.split()[0]
    return ("5.3j" if "M1-M2" in b else "5.3d") if c == "5.3" else c


def metrics(truth, pred, sel=None):
    idx = [i for i in range(len(truth)) if sel is None or sel(truth[i])]
    top1 = np.mean([pred[i] == truth[i] for i in idx])
    by = collections.defaultdict(list)
    for i in idx:
        by[truth[i]].append(pred[i] == truth[i])
    return float(top1), float(np.mean([np.mean(v) for v in by.values()]))


def main():
    import multiprocessing as mp
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    new, stat = {}, collections.Counter()
    with mp.Pool(8) as p:
        for o, st in p.starmap(case_job, list(by.items()), chunksize=1):
            new.update(o); stat.update(st)
    json.dump(new, open(f"{D}/k2_sym_train.json", "w"))
    changed = sum(1 for r in rows if new[f"{r['case']}|{r['lesion_mask_idx']}"] != r["bp_mm"])
    print(f"# K2 좌우 대칭 앵커 보간 — train OOF 스크리닝\n")
    print(f"참조 그래프 보간 통계(케이스×쪽): {dict(sorted(stat.items()))}")
    print(f"bp_mm 이 바뀐 학습 행 {changed}/{len(rows)}\n")
    with mp.Pool(2) as p:
        res = {a: (o, t) for a, o, t in p.map(run_arm, ["base", "sym"])}
    truth = res["base"][1]
    base = {sd: metrics(truth, res["base"][0][sd]) for sd in range(5)}
    symm = {sd: metrics(truth, res["sym"][0][sd]) for sd in range(5)}
    d1 = np.array([symm[s][0] - base[s][0] for s in range(5)]); d2 = np.array([symm[s][1] - base[s][1] for s in range(5)])
    tgt = lambda g: code(g) in ("3.2", "3.3", "3.4", "3.5", "3.6")
    tb = np.mean([metrics(truth, res["base"][0][s], tgt)[0] for s in range(5)])
    ta = np.mean([metrics(truth, res["sym"][0][s], tgt)[0] for s in range(5)])
    ok = d2.mean() >= 0.02 and d1.mean() >= -0.005 and (d2 > 0).sum() >= 4
    b = np.array(list(base.values())).mean(0); a = np.array(list(symm.values())).mean(0)
    print(f"| 팔 | top1 | macro-recall | 표적 3.2~3.6 top1 |\n|---|---|---|---|")
    print(f"| base | {b[0]:.3f} | {b[1]:.3f} | {tb:.3f} |\n| sym | {a[0]:.3f} | {a[1]:.3f} | {ta:.3f} |")
    print(f"\nΔtop1 {d1.mean():+.3f} · Δmacro-recall {d2.mean():+.3f} · 시드별 Δmacro {' '.join(f'{x:+.3f}' for x in d2)} ({(d2 > 0).sum()}/5)")
    print(f"\n**관문 → {'통과 · e2e 진행' if ok else '미달 · e2e 안 건다'}**")
    json.dump(dict(ok=bool(ok), dtop1=float(d1.mean()), dmacro=float(d2.mean())), open(f"{D}/k2_gate.json", "w"))


if __name__ == "__main__":
    main()
