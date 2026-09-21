#!/usr/bin/env python3
"""TAB0 — TabICLv2 대 RF, 환자 단위 LOCO OOF 관문 (2026-09-17 · 사용자 승인).

왜 관문부터인가
    e2e 5시드는 4~5시간이다. 그전에 **기존 112차원 안에 RF 가 회수하지 못한 정보가
    남았는지**만 먼저 묻는다. 못 이기면 여기서 닫고 e2e 를 돌리지 않는다.
    반대로 이겨도 채택이 아니다 — OOF→e2e 괴리를 V3-P·V4-D·R3 에서 세 번 겪었다.
    **이 장치는 통과 기준이 아니라 탈락 기준이다.**

무엇을 재나
    구 학습표 e11_feat_hyb_ov.json(268행·43클래스) 을 c5 와 **똑같이** 112차원으로 만든다
    (c5_location_v2.row_to_vec 을 import · USE_POS=True · 미러증강 → 536행·48클래스).
    환자 단위 leave-one-case-out(214폴드). 미러행은 원본과 **같은 폴드**에 묶어 누수를 막는다.
    평가는 홀드아웃 케이스의 **원본행만** 한다(추론은 원본 방향으로 돈다).
    분류기만 교체한다 — 피처·미러·폴드가 두 모델에서 완전히 동일하다(짝지은 비교).

표적 (FRAC035_weakness_2026-09-17/lesions.csv 감사본에서 정해짐 · 좌우 합산)
    1.9 BA-SCA(학습 2행) · 1.3 VA-PICA(3행) · 3.6 C7-nonBranch(7행) · 3.5 C7-AChA(9행) = 21행

── 판정규칙 (결과 보기 전 고정 · 2026-09-17 14:30 KST) ──────────────────────
 ① 완주: 48클래스 학습·추론이 에러 없이 끝난다. (스모크에서 확인 — 46클래스 many-class 경로)
 ② **주판정**: 표적 21행의 **top-2 정답 수**(3시드 평균)가 RF 보다 **많을 것**. 동수는 미달.
     top-2 를 보는 이유: 제출 구성은 gC(TOPK=2)로 라벨을 둘까지 낸다. 감사본의 "정답 포함률"과
     같은 규약이다. top-1 은 보조로 같이 찍되 판정에 쓰지 않는다.
 ③ **안전**: 전체 268행 top-1 정확도가 RF 대비 **−2.0%p 이내**. 표적을 얻고 전체를 잃으면 기각.
 ②∧③ 을 둘 다 만족할 때만 e2e 5시드로 간다. 하나라도 미달이면 **TabICL 축을 닫는다.**
 보조 기록(판정 미사용): 희소 클래스(학습 ≤5행) 전체 · 클래스별 표 · 추론 시간 · GPU 메모리.
 검정은 하지 않는다 — 21행은 부호검정을 걸 표본이 아니다. 개수와 방향만 본다.
─────────────────────────────────────────────────────────────────────────
"""
import os, sys, time, json, collections
import numpy as np

SC = "/tmp/scratch"
sys.path.insert(0, SC)
import tab_feat

SEEDS = [0, 1, 2]
TARGET_SEG = {"1.9": "BA-SCA", "1.3": "VA-PICA", "3.6": "C7-nonBranch", "3.5": "C7-AChA"}


def seg_of(name):
    """'R-1.9 BA-SCA junction' → '1.9'"""
    return name.split()[0].split("-")[-1]


def loco_folds(groups):
    order = sorted(set(groups))
    for c in order:
        te = np.where(groups == c)[0]
        yield np.where(groups != c)[0], te


def run_rf(X, y, groups, orig, seed):
    from sklearn.ensemble import RandomForestClassifier
    top1, top2 = {}, {}
    for tr, te in loco_folds(groups):
        ev = te[orig[te]]
        if len(ev) == 0:
            continue
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                     class_weight="balanced", random_state=seed, n_jobs=8).fit(X[tr], y[tr])
        p = clf.predict_proba(X[ev]); cls = np.array(clf.classes_)
        o = np.argsort(-p, axis=1)[:, :2]
        for i, r in enumerate(ev):
            top1[r] = cls[o[i, 0]]
            top2[r] = set(cls[o[i, :2]])
    return top1, top2


def run_tabicl(X, y, groups, orig, seed):
    from tabicl import TabICLClassifier
    top1, top2 = {}, {}
    for tr, te in loco_folds(groups):
        ev = te[orig[te]]
        if len(ev) == 0:
            continue
        clf = TabICLClassifier(random_state=seed, device="cuda", n_jobs=1, verbose=False)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[ev]); cls = np.array(clf.classes_)
        o = np.argsort(-p, axis=1)[:, :2]
        for i, r in enumerate(ev):
            top1[r] = cls[o[i, 0]]
            top2[r] = set(cls[o[i, :2]])
    return top1, top2


def score(top1, top2, y, idx):
    a1 = sum(1 for r in idx if top1.get(r) == y[r])
    a2 = sum(1 for r in idx if y[r] in top2.get(r, ()))
    return a1, a2


def main():
    t0 = time.time()
    X, y, groups, orig, rows = tab_feat.build()
    oidx = np.where(orig)[0]
    trcnt = collections.Counter(y[oidx])
    tgt = [r for r in oidx if seg_of(y[r]) in TARGET_SEG]
    rare = [r for r in oidx if trcnt[y[r]] <= 5]
    print(f"[tab0] X {X.shape} · 원본행 {len(oidx)} · 케이스 {len(set(groups))} · "
          f"클래스 {len(set(y))} · 표적 {len(tgt)}행 · 희소(≤5) {len(rare)}행", flush=True)

    res = {}
    for mname, fn in (("RF", run_rf), ("TabICL", run_tabicl)):
        res[mname] = []
        for sd in SEEDS:
            t = time.time()
            t1, t2 = fn(X, y, groups, orig, sd)
            dt = time.time() - t
            res[mname].append((t1, t2, dt))
            a = score(t1, t2, y, oidx); g = score(t1, t2, y, tgt); r = score(t1, t2, y, rare)
            print(f"[tab0] {mname} seed{sd} {dt:6.1f}s · 전체 top1 {a[0]}/{len(oidx)} "
                  f"({a[0]/len(oidx)*100:.1f}%) top2 {a[1]} · 표적 top1 {g[0]}/{len(tgt)} top2 {g[1]}"
                  f" · 희소 top1 {r[0]}/{len(rare)} top2 {r[1]}", flush=True)

    def avg(m, idx, k):
        return float(np.mean([score(t1, t2, y, idx)[k] for t1, t2, _ in res[m]]))

    out = []
    W = out.append
    W("# TAB0 — TabICLv2 대 RF · 환자단위 LOCO OOF 관문")
    W("")
    W(f"규칙은 `tab0_oof.py` 머리말에 결과 보기 전 고정. 시드 {SEEDS} · 폴드 {len(set(groups))}(LOCO) · "
      f"피처 112차원 동일 · 미러증강 동일 · 폴드 동일(짝지은 비교).")
    W("")
    W("## 1) 요약 — 시드 3개 평균 정답 **개수**")
    W("")
    W("| 집합 | 행 | RF top1 | TabICL top1 | RF top2 | TabICL top2 |")
    W("|---|---|---|---|---|---|")
    for nm, idx in (("**표적 4클래스**", tgt), ("희소(학습 ≤5행)", rare), ("전체", oidx)):
        W(f"| {nm} | {len(idx)} | {avg('RF',idx,0):.1f} | {avg('TabICL',idx,0):.1f} | "
          f"{avg('RF',idx,1):.1f} | {avg('TabICL',idx,1):.1f} |")
    W("")
    r2, t2_ = avg("RF", tgt, 1), avg("TabICL", tgt, 1)
    ra, ta = avg("RF", oidx, 0) / len(oidx) * 100, avg("TabICL", oidx, 0) / len(oidx) * 100
    ok2 = t2_ > r2
    ok3 = (ta - ra) >= -2.0
    W("## 2) 판정")
    W("")
    W("| 규칙 | 내용 | 실측 | 충족 |")
    W("|---|---|---|---|")
    W(f"| ② 주판정 | 표적 21행 top-2 정답 수 > RF | RF {r2:.1f} → TabICL {t2_:.1f} "
      f"({t2_-r2:+.1f}) | {'○' if ok2 else '**✗**'} |")
    W(f"| ③ 안전 | 전체 top-1 정확도 −2.0%p 이내 | RF {ra:.1f}% → TabICL {ta:.1f}% "
      f"({ta-ra:+.1f}%p) | {'○' if ok3 else '**✗**'} |")
    W("")
    W(f"**판정 → {'통과 — e2e 5시드로 진행' if (ok2 and ok3) else '미달 — TabICL 축 닫음'}**")
    W("")
    W("천장 확인: 표적 4클래스는 test 검출병변 9개 · val 4개뿐이다(감사본). OOF 에서 이겨도 "
      "e2e 순증의 상한은 test 시드당 +7.5 TP 이고, FP 수지가 그 위에 얹힌다.")
    W("")
    W("## 3) 표적 클래스별 (시드 평균 top-2 정답 수)")
    W("")
    W("| 클래스 | 학습행 | RF | TabICL |")
    W("|---|---|---|---|")
    for c in sorted({y[r] for r in tgt}):
        ii = [r for r in tgt if y[r] == c]
        W(f"| {c} | {len(ii)} | {avg('RF',ii,1):.1f} | {avg('TabICL',ii,1):.1f} |")
    W("")
    W("## 4) 실측 (판정 미사용)")
    W("")
    W(f"- LOCO 214폴드 1회 소요: RF {np.mean([d for _,_,d in res['RF']]):.0f}s · "
      f"TabICL {np.mean([d for _,_,d in res['TabICL']]):.0f}s")
    W(f"- TabICL 체크포인트 105MB · 48클래스는 many-class(계층) 경로로 처리 · GPU 피크 약 1.1GB")
    W(f"- 총 소요 {time.time()-t0:.0f}s")

    D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/V1_vessel_axis"
    open(f"{D}/RESULTS_TAB0.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main()
