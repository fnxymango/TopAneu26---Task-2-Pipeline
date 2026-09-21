#!/usr/bin/env python3
"""E-0 — ICA 전용 분류기(`TOPANEU_ICA_EXPERT`) LOCO OOF 관문 (2026-09-17 · 사용자 승인).

왜 이것만 남았나
    "희소 클래스 의심이면 전용 분류기를 호출하거나 RF 와 평균낸다"는 제안은 대부분 닫힌 축이다 —
    C12(그룹별 전문가) 기각 · DuoRF/K1(확률평균) 닫음 · K7(표 3개 평균) 닫음 ·
    K1 게이팅(애매구간만) 닫음 · conf 게이트(PROJECT_RULES.md 닫힌 목록).
    그런데 `ICA_EXPERT` 는 **c5 에 구현돼 있고 실행 기록이 0건**이다(`OUT_GROW` 와 같은 사례).

왜 C12 의 실패 기전을 피하나
    C12 는 5-way 그룹 판정을 **먼저** 하고 그룹별 전문가에 넘겨서, 1단 오류가 복구 불가능하게
    전파됐다. `_ica_redistribute` 는 다르다 — **ICA 확률 질량 총합을 보존**하고 그 질량을
    ICA 내부 클래스에만 다시 나눈다(c5_location_v2.py:757-781). 그룹 판정을 건드리지 않으므로
    비ICA 행의 판정은 **정의상 불변**이어야 한다. 이걸 규칙 ④ 로 실측 검증한다.
    남는 실패 기전은 C12 의 두 번째, **표본 감소** 하나다: 전문가는 ICA 92행(미러 184)만 본다.

재현
    c5 와 같다. 전문가 = 같은 피처·같은 미러증강으로 ICA 표본(정규식 ^(?:[LR]-)?3\\.)만 재학습.
    추론: q = 전문가 확률 → ICA 인덱스에 mass * (w/Σw) 로 재분배. 질량 보존.
    장치는 TAB0·G-0 와 동일(LOCO 214폴드 · 시드 0,1,2 · 짝지은 비교 · 원본행만 평가).

── 판정규칙 (결과 보기 전 고정 · 2026-09-17 15:25 KST) ──────────────────────
 ② **주판정**: ICA 92행의 **top-2 정답 수**(3시드 평균)가 기준보다 **많을 것**. 동수 미달.
 ③ **안전**: ICA 92행 **top-1** 정답 수가 기준 대비 **−2 이내**.
     top-2 를 얻고 top-1 을 크게 잃는 교환을 막는다 — TabICL 이 정확히 그 방식으로 졌다
     (희소 top-2 +2.6 인데 top-1 −4.3).
 ④ **구현 검증**: 비ICA 176행의 top-1·top-2 판정이 기준과 **완전 동일**할 것.
     질량 보존 가정이 깨지면 구현 오류이므로 판정을 멈추고 원인부터 본다.
 ②∧③∧④ 를 다 만족해야 e2e 로 간다. 하나라도 미달이면 **이 축을 닫는다.**
 보조 기록(판정 미사용): 3.5·3.6 취약 16행 · 클래스별 표.
 검정은 하지 않는다 — 방향과 개수만 본다.
─────────────────────────────────────────────────────────────────────────
"""
import os, re, sys, time, collections
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
SC = "/tmp/scratch"
for p in (f"{R}/code/sblee/nnunet/scripts", f"{R}/code/sblee", f"{R}/code/sblee/nnunet", SC):
    sys.path.insert(0, p)
os.environ.setdefault("TOPANEU_ROOT", R)
import tab_feat

ICA_RE = re.compile(r"^(?:[LR]-)?3\.")
SEEDS = [0, 1, 2]
WEAK = {"3.5", "3.6"}


def loco_folds(groups):
    for c in sorted(set(groups)):
        te = np.where(groups == c)[0]
        yield np.where(groups != c)[0], te


def run(X, y, groups, orig, seed, expert):
    """expert=False 면 기준 RF. True 면 c5 의 _ica_redistribute 와 같은 재분배를 적용."""
    from sklearn.ensemble import RandomForestClassifier
    t1, t2 = {}, {}
    nexp = 0
    for tr, te in loco_folds(groups):
        ev = te[orig[te]]
        if len(ev) == 0:
            continue
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                     class_weight="balanced", random_state=seed, n_jobs=6).fit(X[tr], y[tr])
        cls = np.array(clf.classes_)
        P = clf.predict_proba(X[ev])
        if expert:
            m = np.array([bool(ICA_RE.match(str(t))) for t in y[tr]])
            if m.sum() >= 20 and len(set(y[tr][m])) >= 2:
                nexp += 1
                ex = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                            class_weight="balanced", random_state=seed,
                                            n_jobs=6).fit(X[tr][m], y[tr][m])
                idx = [j for j, c in enumerate(cls) if ICA_RE.match(str(c))]
                if idx:
                    Q = ex.predict_proba(X[ev])
                    qi = {str(c): j for j, c in enumerate(ex.classes_)}
                    for k in range(len(ev)):
                        mass = float(P[k, idx].sum())
                        if mass <= 0:
                            continue
                        w = np.array([Q[k, qi[str(cls[j])]] if str(cls[j]) in qi else 0.0 for j in idx])
                        if w.sum() <= 0:
                            continue
                        P[k, idx] = mass * (w / w.sum())
        o = np.argsort(-P, axis=1)[:, :2]
        for k, r in enumerate(ev):
            t1[r] = cls[o[k, 0]]; t2[r] = set(cls[o[k, :2]])
    return t1, t2, nexp


def score(t1, t2, y, idx):
    return (sum(1 for r in idx if t1.get(r) == y[r]),
            sum(1 for r in idx if y[r] in t2.get(r, ())))


def main():
    t0 = time.time()
    X, y, groups, orig, rows = tab_feat.build()
    oidx = np.where(orig)[0]
    ica = [r for r in oidx if ICA_RE.match(str(y[r]))]
    non = [r for r in oidx if not ICA_RE.match(str(y[r]))]
    weak = [r for r in ica if y[r].split()[0].split("-")[-1] in WEAK]
    print(f"[e0] X {X.shape} · ICA {len(ica)}행 · 비ICA {len(non)}행 · 취약(3.5·3.6) {len(weak)}행 · "
          f"ICA 클래스 {len({y[r] for r in ica})}", flush=True)

    res = {}
    for nm, ex in (("기준", False), ("ICA전문가", True)):
        res[nm] = []
        for sd in SEEDS:
            t = time.time(); t1, t2, ne = run(X, y, groups, orig, sd, ex); dt = time.time() - t
            res[nm].append((t1, t2))
            a = score(t1, t2, y, ica); b = score(t1, t2, y, non); w = score(t1, t2, y, weak)
            print(f"[e0] {nm:9s} seed{sd} {dt:6.1f}s · 전문가발동 {ne}폴드 · "
                  f"ICA top1 {a[0]}/{len(ica)} top2 {a[1]} · 비ICA top1 {b[0]}/{len(non)} · "
                  f"취약 top2 {w[1]}/{len(weak)}", flush=True)

    def avg(nm, idx, k):
        return float(np.mean([score(t1, t2, y, idx)[k] for t1, t2 in res[nm]]))

    # ④ 구현 검증 — 비ICA 판정이 시드별로 완전 동일한가
    same = True; diff = 0
    for (b1, b2), (e1, e2) in zip(res["기준"], res["ICA전문가"]):
        for r in non:
            if b1.get(r) != e1.get(r) or b2.get(r) != e2.get(r):
                same = False; diff += 1

    O = []; W = O.append
    W("# E-0 — ICA 전용 분류기 `TOPANEU_ICA_EXPERT` · LOCO OOF 관문")
    W("")
    W(f"규칙은 `e0_icaexpert.py` 머리말에 결과 보기 전 고정. TAB0·G-0 와 같은 장치 · 시드 {SEEDS} · "
      f"LOCO {len(set(groups))}폴드 · ICA {len(ica)}행 / 비ICA {len(non)}행.")
    W("")
    W("## 1) 시드 3개 평균 정답 개수")
    W("")
    W("| 팔 | ICA top1 | ICA top2 | 취약 3.5·3.6 top2 | 비ICA top1 |")
    W("|---|---|---|---|---|")
    for nm in ("기준", "ICA전문가"):
        W(f"| {nm} | {avg(nm,ica,0):.1f} | **{avg(nm,ica,1):.1f}** | {avg(nm,weak,1):.1f} | "
          f"{avg(nm,non,0):.1f} |")
    W("")
    b1_, e1_ = avg("기준", ica, 0), avg("ICA전문가", ica, 0)
    b2_, e2_ = avg("기준", ica, 1), avg("ICA전문가", ica, 1)
    ok2, ok3, ok4 = e2_ > b2_, (e1_ - b1_) >= -2.0, same
    W("## 2) 판정")
    W("")
    W("| 규칙 | 실측 | 충족 |")
    W("|---|---|---|")
    W(f"| ② ICA top-2 > 기준 | {b2_:.1f} → {e2_:.1f} ({e2_-b2_:+.1f}) | {'○' if ok2 else '**✗**'} |")
    W(f"| ③ ICA top-1 −2 이내 | {b1_:.1f} → {e1_:.1f} ({e1_-b1_:+.1f}) | {'○' if ok3 else '**✗**'} |")
    W(f"| ④ 비ICA 판정 완전 동일 | {'동일' if same else f'**다름 {diff}건**'} | {'○' if ok4 else '**✗**'} |")
    W("")
    W(f"**판정 → {'통과 — e2e 로 진행' if (ok2 and ok3 and ok4) else '미달 — ICA 전용분류기 축 닫음'}**")
    W("")
    W("## 3) ICA 클래스별 (시드평균 top-2)")
    W("")
    W("| 클래스 | 학습행 | 기준 | ICA전문가 |")
    W("|---|---|---|---|")
    for c in sorted({y[r] for r in ica}):
        ii = [r for r in ica if y[r] == c]
        W(f"| {c} | {len(ii)} | {avg('기준',ii,1):.1f} | {avg('ICA전문가',ii,1):.1f} |")
    W("")
    W(f"## 4) 총 {time.time()-t0:.0f}s")
    W("")
    W("남는 실패 기전은 C12 의 **표본 감소** 하나다 — 전문가는 ICA 92행(미러 184)·14클래스만 본다. "
      "C12 의 1단 오류 전파는 질량 보존 설계라 구조적으로 발생하지 않는다(규칙 ④ 로 실측).")
    open(f"{R}/experiments/V1_vessel_axis/RESULTS_E0.md", "w").write("\n".join(O) + "\n")
    print("\n".join(O), flush=True)


if __name__ == "__main__":
    main()
