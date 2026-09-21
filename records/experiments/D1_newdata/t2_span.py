#!/usr/bin/env python3
"""T2 — 분절 걸침 피처 · 추출 + train OOF 스크리닝 (test·val 안 봄).

방추형 오답은 여러 분절에 걸친 큰 병변을 '닿은 곳=접합' 으로 읽는 데서 난다. 현 피처는 혈관별 최단거리·중첩 수라
"병변이 혈관을 따라 얼마나 길게, 몇 분절에 걸쳐 있나" 를 직접 말하지 않는다.
피처 4 (학습표와 같은 GT 병변 · GT 혈관, 낭 복셀은 혈관에서 제외):
  걸친 혈관 수   병변 1mm 외곽 혈관 라벨 중 비율 ≥15% 인 것의 개수
  주축 길이(mm)  병변 주축 방향 투영 폭(5~95 퍼센타일)
  양 끝 같은 혈관 주축 양 끝 15% 복셀 각각의 최근접 혈관(≤3mm) 이 같으면 1 · 다르면 0 · 한쪽 없음 −1
  외곽 구성 엔트로피
미러: 좌우 무관(혈관명 좌우 접두 제거) → 같은 값.
── 관문 (결과 보기 전 고정 · 2026-09-15 · V5 틀) ──
 통과 = 평균 Δmacro-recall ≥ +0.02 ∧ 평균 Δtop1 ≥ −0.005 ∧ Δmacro>0 시드 ≥ 4/5 → e2e 후보(추론 쪽 피처 구현 후 · 사용자 보고)
 참고: 비낭형 top1 · 후순환 top1
"""
import json, os, sys, re, collections
import numpy as np, nibabel as nib
from scipy import ndimage
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D); os.environ.setdefault("TOPANEU_ROOT", R)
ST = np.ones((3, 3, 3), bool)


def case(cid, rows):
    import d9xx_lib as L
    names = L.vessel_dense_names()
    li = nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz")
    loc = np.asanyarray(li.dataobj); sp = np.array(li.header.get_zooms()[:3], float)
    ves = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz").dataobj)
    lab, _ = ndimage.label(loc > 0, structure=ST)
    it = max(1, int(round(1.0 / sp.min())))
    out = {}
    for r in rows:
        m = lab == r["lesion_mask_idx"]; idx = np.argwhere(m)
        lo = np.maximum(idx.min(0) - 8, 0); hi = np.minimum(idx.max(0) + 9, loc.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        mc = m[sl]; v = np.where(mc, 0, ves[sl])
        shell = ndimage.binary_dilation(mc, structure=ST, iterations=it) & ~mc
        sv = v[shell]; sv = sv[sv > 0]
        comp = collections.Counter(re.sub(r"^[RL]-", "", names[int(x)]) for x in sv)
        tot = sum(comp.values())
        fr = np.array([n / tot for n in comp.values()]) if tot else np.array([])
        n_span = int((fr >= 0.15).sum()) if tot else 0
        ent = float(-(fr * np.log(fr)).sum()) if tot else 0.0
        pts = (idx - lo) * sp
        if len(pts) >= 4:
            ax = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, -1]
            proj = pts @ ax
            p5, p15, p85, p95 = np.percentile(proj, [5, 15, 85, 95])
            length = float(p95 - p5)
            vb = v > 0
            if vb.any():
                dv, ind = ndimage.distance_transform_edt(~vb, sampling=sp, return_indices=True)
                def end_label(sel):
                    e = (idx - lo)[sel]
                    d = dv[tuple(e.T)]
                    e = e[d <= 3.0]
                    if not len(e):
                        return None
                    labs = v[tuple(ind[:, e[:, 0], e[:, 1], e[:, 2]])]
                    labs = labs[labs > 0]
                    return re.sub(r"^[RL]-", "", names[int(np.bincount(labs).argmax())]) if labs.size else None
                a, b = end_label(proj <= p15), end_label(proj >= p85)
                same = -1.0 if (a is None or b is None) else float(a == b)
            else:
                same = -1.0
        else:
            length, same = 0.0, -1.0
        out[f"{cid}|{r['lesion_mask_idx']}"] = [float(n_span), length, same, ent]
    return out


def run_arm(arm):
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    F = json.load(open(f"{D}/t2_span_train.json"))
    orig = C5.row_to_vec

    def vec(r, ves_axis, mirror=False):
        v = orig(r, ves_axis, mirror=mirror)
        if arm == "base":
            return v
        a = np.array(F[f"{r['case']}|{r['lesion_mask_idx']}"], float)
        a = np.sign(a) * np.log1p(np.abs(a)); na = np.linalg.norm(a)
        v = np.concatenate([v, a / na * 0.5 if na > 0 else np.zeros(4)]); n = np.linalg.norm(v)
        return v / n if n > 0 else v
    C5.row_to_vec = vec
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    cases = sorted({r["case"] for r in rows}); out = {}
    for sd in range(5):
        os.environ["CLF_SEED"] = str(sd)
        perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
        pred = {}
        for k in range(5):
            m = C5.fit_model([r for r in rows if fold[r["case"]] != k], ax, kind="rf", mirror=True)
            for i, r in enumerate(rows):
                if fold[r["case"]] == k:
                    pred[i] = C5.predict_one(m, r, 0.5)
        out[sd] = [pred[i] for i in range(len(rows))]
    return arm, out


def main():
    import multiprocessing as mp
    from k2_sym import metrics, code
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    feat = {}
    with mp.Pool(6) as p:
        for o in p.starmap(case, list(by.items()), chunksize=1):
            feat.update(o)
    assert len(feat) == len(rows)
    json.dump(feat, open(f"{D}/t2_span_train.json", "w"))
    T = json.load(open(f"{D}/k6_type_feat.json"))
    keys = [f"{r['case']}|{r['lesion_mask_idx']}" for r in rows]
    ns = np.array([T[k]["type"] in (2, 3) for k in keys]); X = np.array([feat[k] for k in keys])
    print("# T2 분절 걸침 피처 — train OOF 스크리닝\n")
    print("| 피처 | 낭형 중앙 | 비낭형 중앙 |\n|---|---|---|")
    for j, nm in enumerate(["걸친 혈관 수", "주축 길이 mm", "양 끝 같은 혈관", "외곽 엔트로피"]):
        print(f"| {nm} | {np.median(X[~ns, j]):.2f} | {np.median(X[ns, j]):.2f} |")
    truth = [r["gt_loc"] for r in rows]
    with mp.Pool(2) as p:
        res = dict(p.map(run_arm, ["base", "t2"]))
    b = {s: metrics(truth, res["base"][s]) for s in range(5)}; t = {s: metrics(truth, res["t2"][s]) for s in range(5)}
    d1 = np.array([t[s][0] - b[s][0] for s in range(5)]); d2 = np.array([t[s][1] - b[s][1] for s in range(5)])
    nsi = np.where(ns)[0]
    acc_ns = lambda arm: np.mean([np.mean([res[arm][s][i] == truth[i] for i in nsi]) for s in range(5)])
    post = lambda g: code(g).split(".")[0] in ("1", "2")
    pb = np.mean([metrics(truth, res["base"][s], post)[0] for s in range(5)]); pt = np.mean([metrics(truth, res["t2"][s], post)[0] for s in range(5)])
    ok = d2.mean() >= 0.02 and d1.mean() >= -0.005 and (d2 > 0).sum() >= 4
    B = np.array(list(b.values())).mean(0); Tt = np.array(list(t.values())).mean(0)
    print("\n| 팔 | top1 | macro-recall | 비낭형 top1 | 후순환 top1 |\n|---|---|---|---|---|")
    print(f"| base | {B[0]:.3f} | {B[1]:.3f} | {acc_ns('base'):.3f} | {pb:.3f} |\n| T2 | {Tt[0]:.3f} | {Tt[1]:.3f} | {acc_ns('t2'):.3f} | {pt:.3f} |")
    print(f"\nΔtop1 {d1.mean():+.3f} · Δmacro {d2.mean():+.3f} · 시드별 {' '.join(f'{x:+.3f}' for x in d2)} ({(d2 > 0).sum()}/5)")
    print(f"\n**관문 → {'통과 · e2e 후보(추론 구현 필요)' if ok else '미달 · e2e 안 건다'}**")
    json.dump(dict(ok=bool(ok)), open(f"{D}/t2_gate.json", "w"))


if __name__ == "__main__":
    main()
