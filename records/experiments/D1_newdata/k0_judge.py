#!/usr/bin/env python3
"""K0 — 병변 단위 짝지은 판정 장치 (2026-09-15 사용자 승인 · 규칙은 아래에 결과 보기 전 고정).

왜: 기준선을 시드만 바꿔도 val MCC 가 0.608~0.699(폭 0.091)로 흔들린다. 분류기 개선은 병변 몇 개 단위라
"test·val 각각 ΔMCC ≥ −0.005" 로는 진짜 이득과 노이즈를 가를 수 없다(V3-P · V3-F 가 test/val 반대로 갈림).

단위: 공식 eval 의 TP 단위와 같은 **(split · 케이스 · GT 클래스)** — test+val 130개. 적중 = 그 클래스가 케이스 예측에 등장.
    (tpcount.py 와 같은 규약) 오답 클래스 = 케이스 예측에 있으나 GT 에 없는 클래스 (FP 단위).

── 판정 규칙 (고정) ────────────────────────────────────────────────────────
 기준 태그 B · 후보 태그 C · 같은 시드 0~4 · 패치필터 후(_pf).
 단위마다 net = (C 적중 시드 수) − (B 적중 시드 수) ∈ [−5, 5].
 n+ = net>0 단위 수 · n− = net<0 단위 수.
 1) 주판정:  단측 부호검정  p = P(Bin(n+ + n−, ½) ≥ n+) < 0.05   ∧  n+ > n−
 2) 안전:    ΔFP(5시드 합, test+val) ≤ ΔTP(5시드 합, test+val) — 새로 붙인 오답 클래스가 새로 맞힌 수를 넘지 않는다
 둘 다 → 채택 후보(7지표는 보조 기록). 1) 불충족 → 미채택. 2) 위반 → 기각.

── 장치 채택 전 영가설 보정 (고정) ─────────────────────────────────────────
 기준선 b1Non 시드 0~9 를 5·5 로 나누는 126가지 × 방향 2 = 252 비교(효과 0).
 이 장치의 거짓 양성률(1)∧2) 통과 비율)이 **≤ 0.07** 이면 장치를 쓴다. 넘으면 쓰지 않고 사용자에게 보고.
 (실제 비교는 같은 시드 번호끼리라 RF 난수가 짝지어져 이 영가설보다 산포가 작다 → 보정은 보수적.)

사용: k0_judge.py null                → 영가설 보정
      k0_judge.py <기준태그> <후보태그>  → 판정 (예: b1Non_pf b1pv_pf)
"""
import json, os, sys, itertools, collections
from math import comb
import numpy as np, nibabel as nib
from concurrent.futures import ProcessPoolExecutor

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
H = f"{R}/experiments/H1_patchfilter"
D = f"{R}/experiments/D1_newdata"
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
NAME = {int(k): v for k, v in S["location_classes"].items()}
_GT = {}
ALPHA = float(os.environ.get("K0_ALPHA", "0.05"))   # 여러 후보를 한꺼번에 올릴 때 사전에 낮춘다(K2: 0.01)
SEEDS = [int(x) for x in os.environ.get("K0_SEEDS", "0,1,2,3,4").split(",")]   # 복제 검증은 5,6,7,8,9


def gt_classes():
    if not _GT:
        for sp in ("test", "val"):
            for c in S["splits"][sp]:
                gp = f"{R}/dataset/TopAneu/location_masks/{c}.nii.gz"
                if os.path.exists(gp):
                    _GT[(sp, c)] = {int(x) for x in np.unique(np.asanyarray(nib.load(gp).dataobj)) if x}
    return _GT


def read_run(args):
    tag, sp, sd = args
    out = {}
    for (s, c), g in gt_classes().items():
        if s != sp:
            continue
        pp = f"{H}/pred/{tag}_{sp}_s{sd}/{c}.nii.gz"
        assert os.path.exists(pp), pp
        out[c] = {int(x) for x in np.unique(np.asanyarray(nib.load(pp).dataobj)) if x}
    return (tag, sp, sd), out


def load(tag, seeds):
    gt_classes()
    jobs = [(tag, sp, sd) for sp in ("test", "val") for sd in seeds]
    with ProcessPoolExecutor(10) as ex:
        return dict(ex.map(read_run, jobs))


def unit_stats(runs, tag, seeds):
    """단위별 적중 시드 수 · 5시드 합 TP · FP."""
    hits = collections.Counter(); tp = fp = 0
    for (sp, c), g in gt_classes().items():
        for sd in seeds:
            pc = runs[(tag, sp, sd)][c]
            for k in g:
                if k in pc:
                    hits[(sp, c, k)] += 1; tp += 1
            fp += len(pc - g)
    return hits, tp, fp


def judge(hb, tpb, fpb, hc, tpc, fpc, units):
    net = {u: hc.get(u, 0) - hb.get(u, 0) for u in units}
    npos = sum(v > 0 for v in net.values()); nneg = sum(v < 0 for v in net.values())
    n = npos + nneg
    p = sum(comb(n, k) for k in range(npos, n + 1)) / 2 ** n if n else 1.0
    dtp, dfp = tpc - tpb, fpc - fpb
    main = p < ALPHA and npos > nneg
    safe = dfp <= dtp
    return dict(npos=npos, nneg=nneg, p=p, dtp=dtp, dfp=dfp, main=main, safe=safe, net=net)


def units():
    return [(sp, c, k) for (sp, c), g in gt_classes().items() for k in g]


def cmd_null():
    runs = load("b1Non_pf", range(10))
    U = units()
    res = []
    for A in itertools.combinations(range(10), 5):
        B = tuple(s for s in range(10) if s not in A)
        # 시드 번호가 다른 두 묶음을 '기준/후보' 로 — 태그는 같고 시드만 다르다
        ha, ta, fa = unit_stats(runs, "b1Non_pf", A)
        hb, tb, fb = unit_stats(runs, "b1Non_pf", B)
        for (x, y) in ((ha, ta, fa), (hb, tb, fb)), ((hb, tb, fb), (ha, ta, fa)):
            r = judge(*x, *y, U)
            res.append((r["main"], r["main"] and r["safe"], r["p"], r["npos"], r["nneg"]))
    n = len(res)
    fpr_main = sum(m for m, *_ in res) / n
    fpr_all = sum(a for _, a, *_ in res) / n
    ps = np.array([p for *_, p, _, _ in [(a, b, c, d, e) for a, b, c, d, e in res]])
    disc = np.array([d + e for *_, d, e in res])
    print(f"# K0 영가설 보정 — 기준선 b1Non_pf 시드 0~9 · 5·5 분할 × 방향 = {n} 비교\n")
    print(f"- 단위 {len(U)} (test+val · 케이스×GT클래스)")
    print(f"- 불일치 단위 수(n+ + n−) 중앙 {np.median(disc):.0f} · 범위 {disc.min()}~{disc.max()}")
    print(f"- p 값 분위: 5% {np.percentile(ps, 5):.3f} · 50% {np.median(ps):.3f}")
    print(f"- 거짓 양성률 · 주판정만 {fpr_main:.3f} · 주판정∧안전 **{fpr_all:.3f}**")
    ok = fpr_all <= 0.07
    print(f"\n**보정 관문(≤0.07) → {'통과 · 장치 사용' if ok else '미달 · 장치 사용 안 함'}**")
    json.dump(dict(n=n, fpr_main=fpr_main, fpr_all=fpr_all, ok=ok), open(f"{D}/k0_null.json", "w"), indent=1)


def cmd_judge(base, cand):
    seeds = SEEDS
    runs = {**load(base, seeds), **load(cand, seeds)}
    U = units()
    hb, tpb, fpb = unit_stats(runs, base, seeds)
    hc, tpc, fpc = unit_stats(runs, cand, seeds)
    r = judge(hb, tpb, fpb, hc, tpc, fpc, U)
    null = json.load(open(f"{D}/k0_null.json")) if os.path.exists(f"{D}/k0_null.json") else None
    print(f"# K0 판정 — `{base}` → `{cand}` (시드 {','.join(map(str, seeds))} · test+val {len(U)}단위)\n")
    if null:
        print(f"장치 영가설 보정: 거짓 양성률 {null['fpr_all']:.3f} → {'사용 가능' if null['ok'] else '⚠ 사용 불가'}\n")
    print(f"- 단위별 적중 순증: 오른 단위 **{r['npos']}** · 내린 단위 **{r['nneg']}** · 단측 부호검정 p = **{r['p']:.4f}**")
    print(f"- 5시드 합: ΔTP {r['dtp']:+d} · ΔFP(오답 클래스) {r['dfp']:+d}")
    for sp in ("test", "val"):
        up = sum(1 for u, v in r["net"].items() if u[0] == sp and v > 0)
        dn = sum(1 for u, v in r["net"].items() if u[0] == sp and v < 0)
        print(f"  · {sp}: 오른 {up} · 내린 {dn}")
    ch = sorted(((v, u) for u, v in r["net"].items() if v), key=lambda x: -abs(x[0]))[:15]
    if ch:
        print("\n| split | 케이스 | 클래스 | 적중 시드 순증 |\n|---|---|---|---|")
        for v, (sp, c, k) in ch:
            print(f"| {sp} | {c[8:]} | {NAME[k]} | {v:+d} |")
    # 참고(판정 미사용 · 2026-09-15 K1 사후 발견): 기준에서 그 split 5시드 내내 오답이 0 이던 클래스에 새로 생긴 오답
    clean_new = 0
    for sp in ("test", "val"):
        fb = collections.Counter(); fc = collections.Counter()
        for (s_, c), g in gt_classes().items():
            if s_ != sp:
                continue
            for sd in seeds:
                for k in runs[(base, sp, sd)][c] - g: fb[k] += 1
                for k in runs[(cand, sp, sd)][c] - g: fc[k] += 1
        clean_new += sum(v for k, v in fc.items() if fb[k] == 0)
    print(f"- 참고(판정 미사용): 기준에서 오답 0 이던 클래스에 새로 생긴 오답 {clean_new} (새로 맞힌 ΔTP {r['dtp']:+d} 대비)")
    verdict = "채택 후보" if (r["main"] and r["safe"]) else ("기각(안전 위반)" if not r["safe"] else "미채택")
    print(f"\n- 주판정(p<{ALPHA} ∧ n+>n−): {'충족' if r['main'] else '미충족'} · 안전(ΔFP ≤ ΔTP): {'충족' if r['safe'] else '위반'}")
    print(f"\n**K0 판정 → {verdict}**")


if __name__ == "__main__":
    if sys.argv[1] == "null":
        cmd_null()
    else:
        cmd_judge(sys.argv[1], sys.argv[2])
