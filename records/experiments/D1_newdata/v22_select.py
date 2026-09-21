#!/usr/bin/env python3
"""V2-2 — 분지 없는 절반 메우기(큐 3단계) + 최종 분할 규칙 선택 · 예측 혈관 점검(큐 4단계).

사용: v22_select.py select   → RESULTS_V21V22.md 에 쓸 표 출력 + v2_rule.json
      v22_select.py pred     → RESULTS_V23.md 에 쓸 표 출력 (v2_rule.json 필요)

── 판정규칙 (결과 보기 전에 고정 · 2026-09-14) ──────────────────────────────
모든 **파라미터는 train 에서만** 뽑는다. val 은 고르는 데만 쓴다. test 는 열지 않는다.
경계의 정답값은 max(t) 가 아니라 **강건한 경계**(앞/뒤 구획 복셀 오분류 최소 t) —
V2-0b 에서 원위 끝의 흩어진 라벨 복셀 하나가 max(t)=1.0 을 만든 사례 때문.

[3단계 · 폴백]
  Pcom(분지 커버리지 ~44%)이 없을 때 C6|C7 경계를 무엇으로 추정할지 train 에서 맞춘다.
  후보 x ∈ {[A1A2], [A1A2,OA], [M1]} 로 gt_c6_cut 을 OLS. C7|terminus 는 A1A2 없을 때
  x ∈ {[M1], [AChA]}. 각 후보의 LOO MAE 를 상수(train 중앙값) LOO MAE 와 비교해
  **0.01 이상 줄이는 최선 후보만 채택**. 못 줄이면 폴백은 상수 백분위.

[선택] 후보 규칙
  C   백분위 (train 중앙값 절단)
  B   분지 그대로 (off=0) · 폴백은 C
  B1  분지 + train 오프셋 (Pcom→C6|C7, A1→C7|terminus 의 train 중앙 차이)
  B2  B1 + 3단계 폴백 (폴백이 하나라도 채택됐을 때만)
  A   jskim 0.40/0.75 — **누수 · 참고로만 표시하고 선택 후보에서 뺀다**
  주 지표 = **val 병변 일관성**(병변이 규칙상 붙는 구획 == 병변 클래스가 요구하는 구획) 정답 수.
  1위와 2위 차이가 val 병변 1개 이하 → val 평균 Dice 가 높은 쪽.
  그래도 |ΔDice| < 0.005 → B2 > B1 > B > C 순(해부학 정의에 가까운 쪽).

[4단계 · 예측 혈관] (다음 실험 V2-B 의 가치 판단용 · V2-A 를 막지 않는다)
  같은 규칙을 **예측 혈관**(vespp_*)에 적용했을 때의 병변 일관성이
  GT 혈관 적용 대비 **−10%p 이내**면 "V2-B 가치 있음", 넘으면 "예측 혈관 품질이 병목".
───────────────────────────────────────────────────────────────────────────
"""
import json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ica_split_lib as IL

D = os.path.dirname(os.path.abspath(__file__))
JS = dict(kind="pct", c1=0.40, c2=0.75, c1_pct=0.40, c2_pct=0.75)


def load(mode):
    return (json.load(open(f"{D}/v21_sides_{mode}.json")),
            json.load(open(f"{D}/v21_lesions_{mode}.json")),
            json.load(open(f"{D}/v21_fails_{mode}.json")))


def lesion_acc(lesions, rule, split=None):
    ok = tot = 0
    for r in lesions:
        if split and r["split"] != split:
            continue
        if "lesion_t" not in r:
            continue
        c1, c2, _ = IL.cuts(rule, r["bt"])
        seg = int(IL.seg_of_t(np.array([r["lesion_t"]]), c1, c2)[0])
        ok += seg == r["expected"]; tot += 1
    return ok, tot


def dice_mean(sides, rule, split):
    vals = []
    for s in sides:
        if s["split"] != split or s.get("inverted"):
            continue
        f = f"{D}/v21_cache/{s['case']}_{s['side']}"
        if not os.path.exists(f + "_t.npy"):
            continue
        t = np.load(f + "_t.npy"); g = np.load(f + "_g.npy")
        c1, c2, _ = IL.cuts(rule, s["bt"])
        p = IL.seg_of_t(t, c1, c2)
        ds = []
        for k in (0, 1, 2):
            a, b = p == k, g == k
            den = a.sum() + b.sum()
            if den:
                ds.append(2 * (a & b).sum() / den)
        if ds:
            vals.append(np.mean(ds))
    return float(np.mean(vals)) if vals else float("nan")


def ols_loo(X, y):
    n = len(y); err = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        A = np.c_[np.ones(m.sum()), X[m]]
        w, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        err.append(abs(w[0] + X[i] @ w[1:] - y[i]))
    A = np.c_[np.ones(n), X]
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.mean(err)), w


def fit_fallback(train_sides, target, cands):
    best = None
    lines = []
    for xs in cands:
        rows = [s for s in train_sides if s.get(target) is not None
                and all(s["bt"].get(k) is not None for k in xs)]
        if len(rows) < 10:
            lines.append(f"| {target} ← {'+'.join(xs)} | {len(rows)} | - | - | 표본 부족 |"); continue
        y = np.array([s[target] for s in rows])
        X = np.array([[s["bt"][k] for k in xs] for s in rows])
        mae, w = ols_loo(X, y)
        c_err = float(np.mean([abs(np.median(np.delete(y, i)) - y[i]) for i in range(len(y))]))
        gain = c_err - mae
        lines.append(f"| {target} ← {'+'.join(xs)} | {len(rows)} | {mae:.4f} | {c_err:.4f} | {gain:+.4f} |")
        if gain >= 0.01 and (best is None or mae < best[0]):
            best = (mae, dict(x=list(xs), b0=float(w[0]), w=[float(v) for v in w[1:]]))
    return (best[1] if best else None), lines


def select():
    sides, lesions, fails = load("gt")
    good = [s for s in sides if not s.get("inverted")]
    tr = [s for s in good if s["split"] == "train"]
    print("# V2-1·V2-2 — 병변 단위 일관성 · 폴백 · 분할 규칙 선택\n")
    print(f"측정 side {len(sides)} (강건 경계 기준 뒤집힘 {len(sides)-len(good)}) · 축 실패 {len(fails)} · "
          f"병변 {len(lesions)} (위치 측정 {sum('lesion_t' in r for r in lesions)})\n")
    for r in lesions:
        if "lesion_t" not in r:
            print(f"  미측정 {r['case']} {r['cls']} — {r.get('why')}")
    c1p = float(np.median([s["gt_c6_cut"] for s in tr if s.get("gt_c6_cut") is not None]))
    c2p = float(np.median([s["gt_c7_cut"] for s in tr if s.get("gt_c7_cut") is not None]))
    o1 = [s["gt_c6_cut"] - s["bt"]["Pcom"] for s in tr if s.get("gt_c6_cut") is not None and s["bt"].get("Pcom") is not None]
    o2 = [s["gt_c7_cut"] - s["bt"]["A1A2"] for s in tr if s.get("gt_c7_cut") is not None and s["bt"].get("A1A2") is not None]
    off1, off2 = float(np.median(o1)), float(np.median(o2))
    print("## 1. train 에서 뽑은 파라미터\n")
    print(f"- 백분위 절단(C) **{c1p:.3f} / {c2p:.3f}**  (jskim 0.400/0.750)")
    print(f"- 분지 오프셋(B1) C6|C7 = t_Pcom **{off1:+.3f}** (n={len(o1)}) · C7|terminus = t_A1 **{off2:+.3f}** (n={len(o2)})\n")

    print("## 2. 3단계 · 폴백 (LOO MAE, train)\n")
    print("| 대상 ← 설명변수 | n | 모델 MAE | 상수 MAE | 개선 |")
    print("|---|---|---|---|---|")
    fb1, l1 = fit_fallback(tr, "gt_c6_cut", [["A1A2"], ["A1A2", "OA"], ["M1"]])
    fb2, l2 = fit_fallback(tr, "gt_c7_cut", [["M1"], ["AChA"]])
    for l in l1 + l2:
        print(l)
    print(f"\n채택: C6|C7 폴백 **{fb1['x'] if fb1 else '없음(상수)'}** · C7|terminus 폴백 **{fb2['x'] if fb2 else '없음(상수)'}**\n")

    base = dict(c1_pct=c1p, c2_pct=c2p)
    rules = {"C": dict(kind="pct", c1=c1p, c2=c2p, **base),
             "B": dict(kind="branch", **base),
             "B1": dict(kind="branch", off1=off1, off2=off2, **base)}
    if fb1 or fb2:
        rules["B2"] = dict(kind="branch", off1=off1, off2=off2, fb1=fb1, fb2=fb2, **base)
    js = dict(JS, c1_pct=0.40, c2_pct=0.75)

    print("## 3. 병변 일관성 · Dice\n")
    print("수작업 경계로 셌을 때가 **이 축이 분류기에 줄 수 있는 정보의 상한**이다.\n")
    man = {sp: (sum(r.get("manual_seg") == r["expected"] for r in lesions if r["split"] == sp and "lesion_t" in r),
                sum(1 for r in lesions if r["split"] == sp and "lesion_t" in r)) for sp in ("train", "val")}
    print("| 규칙 | train 병변 | val 병변 | train Dice | val Dice |")
    print("|---|---|---|---|---|")
    print(f"| 수작업 fine GT (상한) | {man['train'][0]}/{man['train'][1]} = {man['train'][0]/max(man['train'][1],1):.0%} | "
          f"{man['val'][0]}/{man['val'][1]} = {man['val'][0]/max(man['val'][1],1):.0%} | 1.000 | 1.000 |")
    score = {}
    for name, rule in [("A jskim(누수·참고)", js)] + list(rules.items()):
        tr_ok, tr_n = lesion_acc(lesions, rule, "train")
        va_ok, va_n = lesion_acc(lesions, rule, "val")
        dtr, dva = dice_mean(sides, rule, "train"), dice_mean(sides, rule, "val")
        print(f"| {name} | {tr_ok}/{tr_n} = {tr_ok/max(tr_n,1):.0%} | {va_ok}/{va_n} = {va_ok/max(va_n,1):.0%} | "
              f"{dtr:.3f} | {dva:.3f} |")
        if name in rules:
            score[name] = (va_ok, dva)

    order = ["B2", "B1", "B", "C"]
    ranked = sorted(score, key=lambda k: (-score[k][0], -score[k][1], order.index(k)))
    top = ranked[0]
    if len(ranked) > 1:
        a, b = ranked[0], ranked[1]
        if score[a][0] - score[b][0] <= 1:
            if abs(score[a][1] - score[b][1]) < 0.005:
                top = min((a, b), key=order.index)
            else:
                top = max((a, b), key=lambda k: score[k][1])
    print(f"\n## 4. 선택\n\n순위(val 병변 → val Dice): " +
          " > ".join(f"{k}({score[k][0]}, {score[k][1]:.3f})" for k in ranked))
    print(f"\n**→ 채택 규칙: {top}**")

    print("\n## 5. 채택 규칙의 병변 클래스별 결과 (train+val)\n")
    print("| 클래스 | 요구 구획 | 규칙 정답 | 수작업 정답 |")
    print("|---|---|---|---|")
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in lesions:
        if "lesion_t" not in r:
            continue
        c1, c2, _ = IL.cuts(rules[top], r["bt"])
        seg = int(IL.seg_of_t(np.array([r["lesion_t"]]), c1, c2)[0])
        a = agg[r["key"]]; a[0] += 1; a[1] += seg == r["expected"]; a[2] += r.get("manual_seg") == r["expected"]
    nm = {0: "C6", 1: "C7", 2: "terminus"}
    for k in sorted(agg):
        n, ok, mo = agg[k]
        from v21_lesion import EXPECT
        print(f"| {k} | {nm[EXPECT[k]]} | {ok}/{n} | {mo}/{n} |")

    json.dump(dict(name=top, rule=rules[top], params=dict(c1_pct=c1p, c2_pct=c2p, off1=off1, off2=off2,
                                                          fb1=fb1, fb2=fb2),
                   val_score={k: list(v) for k, v in score.items()}),
              open(f"{D}/v2_rule.json", "w"), indent=1)


def pred():
    rj = json.load(open(f"{D}/v2_rule.json"))
    rule = rj["rule"]
    gs, gl, gf = load("gt")
    ps, pl, pf = load("pred")
    print(f"# V2-3 — 예측 혈관에서 규칙이 버티는가 (규칙 {rj['name']})\n")
    print(f"예측 혈관 side {len(ps)} · 축 실패 {len(pf)} (GT 혈관은 {len(gf)})\n")
    for f in pf:
        print(f"  축 실패 {f['case']} {f['side']} — {f['why']}")
    cov = lambda ss, b: sum(s["bt"].get(b) is not None for s in ss) / max(len(ss), 1)
    print("\n| 분지 커버리지 | GT 혈관 | 예측 혈관 |\n|---|---|---|")
    for b in ("OA", "Pcom", "AChA", "A1A2", "M1"):
        print(f"| {b} | {cov(gs, b):.0%} | {cov(ps, b):.0%} |")
    print("\n| | train 병변 | val 병변 | 합계 |\n|---|---|---|---|")
    res = {}
    for name, les in (("GT 혈관 + 규칙", gl), ("예측 혈관 + 규칙", pl)):
        a = lesion_acc(les, rule, "train"); b = lesion_acc(les, rule, "val")
        tot = (a[0] + b[0], a[1] + b[1]); res[name] = tot
        print(f"| {name} | {a[0]}/{a[1]} | {b[0]}/{b[1]} | {tot[0]}/{tot[1]} = {tot[0]/max(tot[1],1):.0%} |")
    g = res["GT 혈관 + 규칙"]; p = res["예측 혈관 + 규칙"]
    gap = p[0] / max(p[1], 1) - g[0] / max(g[1], 1)
    print(f"\n예측 − GT = **{gap*100:+.1f}%p**  (측정된 병변 수가 다르면 비율로 비교한다)")
    print(f"\n**판정 → {'V2-B 가치 있음 (−10%p 이내)' if gap >= -0.10 else '예측 혈관 품질이 병목 — V2-B 전에 혈관 쪽 선결'}**")


if __name__ == "__main__":
    {"select": select, "pred": pred}[sys.argv[1]]()
