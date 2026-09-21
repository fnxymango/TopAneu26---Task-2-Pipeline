#!/usr/bin/env python3
"""V5 스크리닝 — 새 피처 묶음 F1(돌출 방향) · F2(굵기) · F3(뼈, CT) 를 train OOF 로만 거른다. test·val 안 봄.

분류기는 e2e 와 **같은 코드**(c5_location_v2.fit_model · predict_one · USE_POS · τ0.5 · β0.5 · 미러)를 쓰고,
row_to_vec 뒤에 블록을 붙인다 — c5 의 추가 블록 규약 그대로(블록 단위 정규화 × 가중 0.5 후 이어붙이고 전체 재정규화).
미러 행: F1 x 성분 부호 반전. F2·F3 는 그대로.
CV: **케이스 단위** 5겹(시드마다 케이스 순서를 섞음) × 시드 0~4. deep_ica 의 행 단위 KFold 는 같은 케이스 병변이 갈라질 수 있어 쓰지 않는다.

── 판정 관문 (결과 보기 전 고정 · 2026-09-15) ───────────────────────────────
 팔: base · +F1 · +F2 · +F3 · +F1F2F3
 통과 = 5시드 평균 Δmacro-recall ≥ +0.02  ∧  평균 Δtop1 ≥ −0.005  ∧  Δmacro-recall > 0 인 시드 ≥ 4/5   (base 대비 · 같은 폴드)
 통과한 팔만 e2e 후보로 올린다(추론 쪽 피처 = 예측 혈관 · 검출 blob · 영상에서 다시 구현 필요).
 표적 구간 정확도(F1→ICA 3.2~3.7 · F2→MCA·ACA·후순환 · F3→3.1·3.3)는 참고로만 적고 판정에 쓰지 않는다.
 ⚠ 8/26 q13 크기 피처는 OOF 에서 오르고 val 에서 떨어졌다(−0.014). 이 관문은 스크리닝일 뿐이다.
"""
import json, os, sys, re, collections
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts")
os.environ.setdefault("TOPANEU_ROOT", R)
W = 0.5
ARMS = {"base": (), "F1": ("f1",), "F2": ("f2",), "F3": ("f3",), "F123": ("f1", "f2", "f3")}


def code(nm):
    b = re.sub(r"^[RL]-", "", nm); c = b.split()[0]
    return ("5.3j" if "M1-M2" in b else "5.3d") if c == "5.3" else c


def load_rows():
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    f12 = json.load(open(f"{D}/v5_feat_train.json")); f3 = json.load(open(f"{D}/v5_bone_train.json"))
    for r in rows:
        k = f"{r['case']}|{r['lesion_mask_idx']}"
        r["_v5"] = dict(f1=f12[k]["f1"], f2=f12[k]["f2"], f3=f3[k])
    return rows


def patch(C5, blocks):
    orig = C5._V5_ORIG if hasattr(C5, "_V5_ORIG") else C5.row_to_vec
    C5._V5_ORIG = orig

    def vec(r, ves_axis, mirror=False):
        v = orig(r, ves_axis, mirror=mirror)
        if not blocks or "_v5" not in r:
            return v
        for b in blocks:
            a = np.array(r["_v5"][b], dtype=float)
            if b == "f1" and mirror:
                a = a.copy(); a[0] = -a[0]
            if b in ("f2", "f3"):
                a = np.sign(a) * np.log1p(np.abs(a))       # mm 크기 차이를 눌러 블록 안 균형
            na = np.linalg.norm(a)
            v = np.concatenate([v, a / na * W if na > 0 else np.zeros_like(a)])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    C5.row_to_vec = vec


def run_arm(arm):
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    patch(C5, ARMS[arm])
    ves_axis = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    rows = load_rows()
    cases = sorted({r["case"] for r in rows})
    out = {}
    for sd in range(5):
        os.environ["CLF_SEED"] = str(sd)
        rng = np.random.default_rng(sd); perm = list(rng.permutation(cases))
        fold = {c: i % 5 for i, c in enumerate(perm)}
        pred = {}
        for k in range(5):
            tr = [r for r in rows if fold[r["case"]] != k]
            m = C5.fit_model(tr, ves_axis, kind="rf", mirror=True)
            for i, r in enumerate(rows):
                if fold[r["case"]] == k:
                    pred[i] = C5.predict_one(m, r, 0.5)
        out[sd] = [pred[i] for i in range(len(rows))]
    return arm, out


def metrics(rows, pred, sel=None):
    idx = range(len(rows)) if sel is None else [i for i in range(len(rows)) if sel(rows[i]["gt_loc"])]
    idx = list(idx)
    if not idx:
        return float("nan"), float("nan")
    top1 = np.mean([pred[i] == rows[i]["gt_loc"] for i in idx])
    by = collections.defaultdict(list)
    for i in idx:
        by[rows[i]["gt_loc"]].append(pred[i] == rows[i]["gt_loc"])
    return float(top1), float(np.mean([np.mean(v) for v in by.values()]))


def main():
    import multiprocessing as mp
    rows = load_rows()
    with mp.Pool(5) as p:
        res = dict(p.map(run_arm, list(ARMS)))
    json.dump(res, open(f"{D}/v5_screen_pred.json", "w"))
    TGT = {"F1": lambda g: code(g) in ("3.2", "3.3", "3.4", "3.5", "3.6", "3.7"),
           "F2": lambda g: code(g).split(".")[0] in ("1", "2", "4", "5"),
           "F3": lambda g: code(g) in ("3.1", "3.3"),
           "F123": None}
    n_ct = sum(r["_v5"]["f3"][2] == 1 for r in rows)
    print(f"# V5 스크리닝 — train OOF (케이스 5겹 × 시드5) · 271병변 · CT {n_ct}\n")
    print("관문: 평균 Δmacro-recall ≥ +0.02 ∧ 평균 Δtop1 ≥ −0.005 ∧ Δmacro-recall>0 시드 ≥ 4/5\n")
    base = {sd: metrics(rows, res["base"][sd]) for sd in range(5)}
    b = np.array(list(base.values())).mean(0)
    print(f"base  top1 {b[0]:.3f} · macro-recall {b[1]:.3f}\n")
    print("| 팔 | top1 | Δtop1 | macro-recall | Δmacro | Δmacro>0 시드 | 표적 구간 top1 (base→팔) | 관문 |\n|---|---|---|---|---|---|---|---|")
    verdict = {}
    for arm in ARMS:
        if arm == "base":
            continue
        mm = {sd: metrics(rows, res[arm][sd]) for sd in range(5)}
        d1 = np.array([mm[sd][0] - base[sd][0] for sd in range(5)])
        d2 = np.array([mm[sd][1] - base[sd][1] for sd in range(5)])
        a = np.array(list(mm.values())).mean(0)
        ok = d2.mean() >= 0.02 and d1.mean() >= -0.005 and (d2 > 0).sum() >= 4
        verdict[arm] = bool(ok)
        t = TGT[arm]
        tg = ""
        if t is not None:
            tb = np.mean([metrics(rows, res["base"][sd], t)[0] for sd in range(5)])
            ta = np.mean([metrics(rows, res[arm][sd], t)[0] for sd in range(5)])
            tg = f"{tb:.3f}→{ta:.3f}"
        print(f"| {arm} | {a[0]:.3f} | {d1.mean():+.3f} | {a[1]:.3f} | {d2.mean():+.3f} | {(d2 > 0).sum()}/5 | {tg} | {'**통과**' if ok else '미달'} |")
    print("\n시드별 Δmacro-recall:")
    for arm in ARMS:
        if arm != "base":
            print(f"  {arm}: " + " ".join(f"{metrics(rows, res[arm][sd])[1] - base[sd][1]:+.3f}" for sd in range(5)))
    json.dump(verdict, open(f"{D}/v5_screen_verdict.json", "w"))
    print(f"\n**e2e 후보: {', '.join(a for a, v in verdict.items() if v) or '없음'}**")


if __name__ == "__main__":
    main()
