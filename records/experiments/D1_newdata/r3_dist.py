#!/usr/bin/env python3
"""R3 — 접합 클래스 거리 상한 규칙 스크리닝 (train OOF 만 · test·val 안 봄).

R1-A 는 곁가지가 **아예 없을 때만** 발동해 test 에서 1단위밖에 안 움직였다(미채택).
R3 는 "곁가지는 있는데 병변이 그 **기시부 분기점에서 멀다**" 를 대상으로 넓힌다.
근거: 최대 오답 쌍이 3.5→3.4 · 3.6→3.4 · 3.3→3.2 로, **접합이 아닌 병변을 접합으로 부르는** 방향에 몰려 있다.

거리는 반드시 **추론 조건**(예측 혈관 + vespp 그래프)에서 잰다 — 학습표(GT 혈관) 거리로 τ 를 잡으면
추론에서 발동하지 않는다(POSTMORTEM 3 · V1-B 실패). 그래서 `c10_feat_train_predves_NEW.json` 의 bp_mm 을 쓴다.
대상 클래스 → 분기점 쌍:
  3.2=(ICA-C6-C7, OA) · 3.4=(ICA-C6-C7, Pcom) · 3.5=(ICA-C6-C7, AChA) · 1.3=(VA, PICA)
  1.7=(BA, AICA) · 1.9=(BA, SCA) · 4.1=(A1A2, Acom)
규칙: 1등이 위 접합 클래스이고 그 쌍의 예측조건 거리가 τmm 초과면 1등을 빼고 다음 후보로 내린다(거리 없음=∞ 도 대상).

── 관문 (결과 보기 전 고정 · 2026-09-16) ────────────────────────────────
 어떤 τ ∈ {4, 6, 8, 10, 12} 에서 세 조건을 다 만족하면 e2e(K0) 후보:
   ① 살아남 ≥ 2 × 새로 틀림   ② 시드당 발동(1등이 바뀐 병변) ≥ 5   ③ 5시드 합 순증 ≥ +10
 τ 는 train OOF 에서만 고른다(조건을 만족하는 τ 중 순증이 가장 큰 것). 아무 τ 도 못 넘으면 R3 를 닫는다.
 ②는 R1-A 의 실패(발동 test 1단위 → 두 집합 불일치)에서 온 조건이다.
"""
import json, os, sys, re, collections
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D)
os.environ.setdefault("TOPANEU_ROOT", R)
TAUS = (4.0, 6.0, 8.0, 10.0, 12.0)
PAIR = {"3.2": ("ICA-C6-C7", "OA"), "3.4": ("ICA-C6-C7", "Pcom"), "3.5": ("ICA-C6-C7", "AChA"),
        "1.3": ("VA", "PICA"), "1.7": ("BA", "AICA"), "1.9": ("BA", "SCA"), "4.1": ("A1A2", "Acom")}


def code(nm):
    return re.sub(r"^[RL]-", "", str(nm)).split()[0]


def side(nm):
    m = re.match(r"^([RL])-", str(nm)); return m.group(1) if m else None


def pair_index():
    import c5_location_v2 as C5
    return {frozenset(p): i for i, p in enumerate(C5.JUNCTION_PAIRS)}


def bp_lookup(row, cls_name, pidx):
    """1등 클래스가 요구하는 분기점까지의 예측조건 거리(mm). 못 찾으면 inf."""
    c, sd = code(cls_name), side(cls_name)
    pr = PAIR.get(c)
    if pr is None:
        return None
    def full(v):
        if v in ("Acom",):
            return v
        return f"{sd}-{v}" if sd else v
    for cand in ({full(pr[0]), full(pr[1])},
                 {f"R-{pr[0]}" if pr[0] != "Acom" else pr[0], f"R-{pr[1]}" if pr[1] != "Acom" else pr[1]},
                 {f"L-{pr[0]}" if pr[0] != "Acom" else pr[0], f"L-{pr[1]}" if pr[1] != "Acom" else pr[1]}):
        k = pidx.get(frozenset(cand))
        if k is not None:
            v = row["bp_mm"][k]
            return float("inf") if v is None else float(v)
    return None


def job(sd):
    os.environ["CLF_SEED"] = str(sd)
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    cases = sorted({r["case"] for r in rows})
    perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
    top = [[] for _ in rows]
    for k in range(5):
        m = C5.fit_model([r for r in rows if fold[r["case"]] != k], ax, kind="rf", mirror=True)
        for i, r in enumerate(rows):
            if fold[r["case"]] == k:
                nm, _, _ = C5.predict_ranked(m, r, 0.5)
                top[i] = [str(x) for x in nm[:4]] if nm is not None and len(nm) else []
    return sd, top


def main():
    from multiprocessing import Pool
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    truth = [r["gt_loc"] for r in rows]
    prd = {(r["case"], r.get("lesion_mask_idx")): r for r in json.load(open(f"{A}/c10_feat_train_predves_NEW.json"))}
    pidx = pair_index()
    with Pool(5) as p:
        res = p.map(job, range(5))
    stat = {t: collections.Counter() for t in TAUS}
    dists = []
    for sd, top in res:
        for i, r in enumerate(rows):
            if not top[i]:
                continue
            t1 = top[i][0]
            q = prd.get((r["case"], r.get("lesion_mask_idx")))
            if q is None:
                continue
            d = bp_lookup(q, t1, pidx)
            if d is None:
                continue
            if sd == 0:
                dists.append((code(t1), d, t1 == truth[i]))
            for t in TAUS:
                if d <= t:
                    continue
                nxt = next((x for x in top[i][1:] if x != t1), None)
                if nxt is None:
                    continue
                ok_old, ok_new = t1 == truth[i], nxt == truth[i]
                s = stat[t]
                s["발동"] += 1; s["살아남"] += (not ok_old) and ok_new; s["새로틀림"] += ok_old and (not ok_new)
    print("# R3 — 접합 클래스 거리 상한 규칙 스크리닝 (train OOF · 시드 0~4 · 규칙은 r3_dist.py 머리말 고정)\n")
    print("참고 · 시드0 에서 1등이 접합 클래스였던 병변의 예측조건 분기점 거리 (정답/오답별 중앙값)")
    for c in sorted({c for c, _, _ in dists}):
        ok = [d for cc, d, o in dists if cc == c and o and np.isfinite(d)]
        ng = [d for cc, d, o in dists if cc == c and not o and np.isfinite(d)]
        print(f"  {c}: 정답 {np.median(ok):.1f}mm(n={len(ok)}) · 오답 {np.median(ng):.1f}mm(n={len(ng)})" if ok and ng else
              f"  {c}: 정답 n={len(ok)} · 오답 n={len(ng)}")
    print("\n| τ(mm) | 발동(5시드 합) | 시드당 발동 | 살아남 | 새로 틀림 | 순증 | 판정 |\n|---|---|---|---|---|---|---|")
    best, out = None, {}
    for t in TAUS:
        s = stat[t]; fx, br, n = s["살아남"], s["새로틀림"], s["발동"]
        ok = (fx >= 2 * br) and (n / 5 >= 5) and (fx - br >= 10)
        out[str(t)] = dict(n=n, fixed=fx, broken=br, ok=bool(ok))
        print(f"| {t:.0f} | {n} | {n/5:.1f} | {fx} | {br} | {fx-br:+d} | {'통과' if ok else '미달'} |")
        if ok and (best is None or fx - br > stat[best]["살아남"] - stat[best]["새로틀림"]):
            best = t
    print(f"\n**관문 → {'통과 · τ=' + f'{best:.0f}mm 로 e2e(K0)' if best else '미달 · R3 닫음'}**")
    json.dump(dict(tau=best, per_tau=out), open(f"{D}/r3_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
