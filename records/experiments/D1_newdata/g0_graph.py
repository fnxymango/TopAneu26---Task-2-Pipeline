#!/usr/bin/env python3
"""G-0 — 혈관 연결관계 피처 블록 관문 (2026-09-17 · 사용자 승인 · GNN 재개 후 첫 단계).

왜 GNN 이 아니라 이걸 먼저 하나
    GNN 은 그래프 위에서 **관계**를 학습한다. 관계가 RF 에 붙였을 때 아무것도 안 주면
    GNN 도 못 준다 — 같은 정보를 파라미터만 더 써서 읽는 것이기 때문이다.
    제안서가 "필수 대조군: 연결관계를 사용하지 않는 모델과 비교" 라고 한 그 대조를
    **싼 방향으로 먼저** 돌린다. 며칠 대신 반나절이다.

측정으로 확인된 전제 (2026-09-17 · 결과 보기 전 기록)
    표적 4클래스 중 3.5 AChA 는 **정의 곁가지가 그래프에 없다**(L-3.5 정의 분기점 0/4행 ·
    곁가지 거리 중앙 3.58mm · 잘 맞히는 3.4 Pcom 은 0.86mm). 1.9·1.3 은 노드는 있으나
    학습행이 1~2개다. 그래서 **이 관문의 사전 기대는 낮다.** 그럼에도 도는 이유는
    분기점의 23%(2혈관 34쌍 밖 18% + 3혈관 이상 5%)가 현재 피처에 안 들어가기 때문이다.

무엇을 붙이나 — 10차원 (H3-block 이 148차원에서 죽었으므로 최소로 잡는다)
    [관계형 5] 지역 차수 · 주혈관 인접 차수 · 주혈관 노드 수 · 최근접노드가 주혈관 포함 · 3혈관 노드 수
    [비관계형 5] 최근접 노드 거리 · 최근접 3혈관 노드 거리 · 최근접 34쌍밖 노드 거리 · 5mm 내 노드 수 · 10mm 내 노드 수
    좌표는 학습표의 `_cen`(병변 중심, mm). c5 의 bp_mm 은 병변 **표면**에서 재므로
    등가반경 r_eq 를 빼서 근사 보정한다(`ds = max(0, dc - r_eq)`). 이 근사는 절대거리에만
    영향을 주고 관계형 5개에는 영향이 없다.
    그래프는 `_c4_bpgraph/all_ref`(GT 혈관) — 학습표 bp_mm 과 같은 출처다.
    미러증강 시 노드 `classes` 와 주혈관 판정도 c5.mirror_name 으로 같이 뒤집는다.

── 판정규칙 (결과 보기 전 고정 · 2026-09-17 15:40 KST) ──────────────────────
 TAB0 와 **같은 장치**(LOCO 214폴드 · 시드 0,1,2 · 표적 21행 · 짝지은 비교)를 쓴다.
 ② **주판정**: 표적 21행 top-2 정답 수(3시드 평균)가 기준(블록 없음)보다 **많을 것**. 동수 미달.
 ③ **안전**: 전체 268행 top-1 정확도가 기준 대비 **−2.0%p 이내**.
 ④ **간선 기여 대조**(제안서 요구): 관계형 5차원을 뺀 **비관계형만 5차원** 팔도 같이 돈다.
     전체팔이 이겼는데 비관계형팔도 똑같이 이기면 **연결관계의 기여가 아니다** → GNN 근거 없음.
 ②∧③ 을 만족하고 ④ 에서 관계형이 비관계형보다 나을 때만 GNN 제작으로 간다.
 하나라도 미달이면 **GNN 축을 닫는다.**
 검정은 하지 않는다 — 21행은 부호검정을 걸 표본이 아니다. 개수와 방향만 본다.
─────────────────────────────────────────────────────────────────────────
"""
import os, sys, json, time, collections
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
SC = "/tmp/scratch"
for p in (f"{R}/code/sblee/nnunet/scripts", f"{R}/code/sblee", f"{R}/code/sblee/nnunet", SC):
    sys.path.insert(0, p)
os.environ.setdefault("TOPANEU_ROOT", R)
import c5_location_v2 as c5
import tab_feat

BPDIR = f"{R}/experiments/_c4_bpgraph/all_ref"
PAIRSET = {frozenset(p) for p in c5.JUNCTION_PAIRS}
SEEDS = [0, 1, 2]
TARGET_SEG = {"1.9", "1.3", "3.6", "3.5"}
REL = [0, 1, 2, 3, 4]      # 관계형 5
NONREL = [5, 6, 7, 8, 9]   # 비관계형 5

_cache = {}


def nodes_of(case):
    if case not in _cache:
        p = f"{BPDIR}/{case}.json"
        d = json.load(open(p)) if os.path.exists(p) else {"nodes": [], "spacing": [1, 1, 1]}
        _cache[case] = (d.get("nodes", []), d.get("spacing", [1, 1, 1]))
    return _cache[case]


def graph_vec(row, mirror):
    """10차원. mirror=True 면 노드 classes 와 주혈관 판정을 뒤집는다(c5.row_to_vec 과 같은 규약)."""
    nodes, spacing = nodes_of(row["case"])
    v = np.zeros(10)
    if not nodes:
        v[5:8] = 99.0   # 거리 3개는 '없음'을 큰 값으로
        return v
    mn = (lambda n: c5.mirror_name(n)) if mirror else (lambda n: n)
    cen = np.array(row["_cen"], dtype=float)
    voxvol = float(np.prod(spacing))
    r_eq = (3.0 * row["n_vox"] * voxvol / (4.0 * np.pi)) ** (1.0 / 3.0)

    # 주혈관 = 병변에서 가장 가까운 혈관 (dist_mm 최소). 미러면 이름을 뒤집는다.
    prim = None
    if row["dist_mm"]:
        prim = mn(min(row["dist_mm"].items(), key=lambda kv: kv[1])[0])

    ds_all, ds_multi, ds_out = [], [], []
    near5 = near10 = 0
    cls10 = set(); prim_nodes10 = 0; multi10 = 0
    best_d, best_has_prim = 1e9, 0.0
    adj = set()   # 케이스 전체에서 주혈관과 한 노드를 공유하는 혈관들
    for nd in nodes:
        if not nd.get("valid", True):
            continue
        cls = [mn(c) for c in nd["classes"]]
        d = float(np.linalg.norm(np.array(nd["centroid_mm"], dtype=float) - cen))
        ds = max(0.0, d - r_eq)
        ds_all.append(ds)
        if len(cls) >= 3:
            ds_multi.append(ds)
        if not (len(cls) == 2 and frozenset(cls) in PAIRSET):
            ds_out.append(ds)
        if prim is not None and prim in cls:
            adj.update(c for c in cls if c != prim)
        if ds <= 5.0:
            near5 += 1
        if ds <= 10.0:
            near10 += 1
            cls10.update(cls)
            if len(cls) >= 3:
                multi10 += 1
            if prim is not None and prim in cls:
                prim_nodes10 += 1
        if ds < best_d:
            best_d = ds
            best_has_prim = 1.0 if (prim is not None and prim in cls) else 0.0

    v[0] = len(cls10)          # 지역 차수 — 10mm 안에 나타나는 혈관 종류 수
    v[1] = len(adj)            # 주혈관 인접 차수 (케이스 전체)
    v[2] = prim_nodes10        # 주혈관을 포함하는 10mm 내 노드 수
    v[3] = best_has_prim       # 최근접 노드가 주혈관을 포함하나
    v[4] = multi10             # 10mm 내 3혈관 이상 노드 수
    v[5] = min(ds_all) if ds_all else 99.0
    v[6] = min(ds_multi) if ds_multi else 99.0
    v[7] = min(ds_out) if ds_out else 99.0
    v[8] = near5
    v[9] = near10
    return v


def build_arms():
    X, y, g, orig, rows = tab_feat.build()
    G = np.zeros((len(X), 10))
    i = 0
    for r in rows:
        G[i] = graph_vec(r, False); i += 1
        G[i] = graph_vec(r, True); i += 1
    assert i == len(X)
    return X, y, g, orig, rows, G


def loco_folds(groups):
    for c in sorted(set(groups)):
        te = np.where(groups == c)[0]
        yield np.where(groups != c)[0], te


def run_rf(X, y, groups, orig, seed):
    from sklearn.ensemble import RandomForestClassifier
    t1, t2 = {}, {}
    for tr, te in loco_folds(groups):
        ev = te[orig[te]]
        if len(ev) == 0:
            continue
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                     class_weight="balanced", random_state=seed, n_jobs=8).fit(X[tr], y[tr])
        p = clf.predict_proba(X[ev]); cls = np.array(clf.classes_)
        o = np.argsort(-p, axis=1)[:, :2]
        for k, r in enumerate(ev):
            t1[r] = cls[o[k, 0]]; t2[r] = set(cls[o[k, :2]])
    return t1, t2


def score(t1, t2, y, idx):
    return (sum(1 for r in idx if t1.get(r) == y[r]),
            sum(1 for r in idx if y[r] in t2.get(r, ())))


def main():
    t0 = time.time()
    X, y, groups, orig, rows, G = build_arms()
    oidx = np.where(orig)[0]
    tgt = [r for r in oidx if y[r].split()[0].split("-")[-1] in TARGET_SEG]
    trcnt = collections.Counter(y[oidx])
    rare = [r for r in oidx if trcnt[y[r]] <= 5]
    print(f"[g0] X {X.shape} · 그래프블록 {G.shape} · 표적 {len(tgt)}행 · 희소 {len(rare)}행", flush=True)
    nz = (G != 0).mean(0)
    print(f"[g0] 블록 비영 비율: " + " ".join(f"{v:.2f}" for v in nz), flush=True)

    ARMS = {"기준(112)": X,
            "전체블록(122)": np.hstack([X, G]),
            "비관계형만(117)": np.hstack([X, G[:, NONREL]])}
    res = {}
    for nm, XX in ARMS.items():
        res[nm] = []
        for sd in SEEDS:
            t = time.time(); a = run_rf(XX, y, groups, orig, sd); dt = time.time() - t
            res[nm].append(a)
            s_all = score(*a, y, oidx); s_t = score(*a, y, tgt); s_r = score(*a, y, rare)
            print(f"[g0] {nm:14s} seed{sd} {dt:6.1f}s · 전체 top1 {s_all[0]} ({s_all[0]/len(oidx)*100:.1f}%) "
                  f"top2 {s_all[1]} · 표적 top1 {s_t[0]}/{len(tgt)} top2 {s_t[1]} · 희소 top2 {s_r[1]}/{len(rare)}",
                  flush=True)

    def avg(nm, idx, k):
        return float(np.mean([score(t1, t2, y, idx)[k] for t1, t2 in res[nm]]))

    O = []; W = O.append
    W("# G-0 — 혈관 연결관계 피처 블록 · LOCO OOF 관문 (GNN 전단계)")
    W("")
    W(f"규칙은 `g0_graph.py` 머리말에 결과 보기 전 고정. TAB0 와 같은 장치 · 시드 {SEEDS} · "
      f"LOCO {len(set(groups))}폴드 · 표적 {len(tgt)}행.")
    W("")
    W("## 1) 세 팔 — 시드 3개 평균 정답 개수")
    W("")
    W("| 팔 | 차원 | 표적 top1 | 표적 top2 | 희소 top2 | 전체 top1 | 전체 top1 % |")
    W("|---|---|---|---|---|---|---|")
    for nm in ARMS:
        W(f"| {nm} | {ARMS[nm].shape[1]} | {avg(nm,tgt,0):.1f} | **{avg(nm,tgt,1):.1f}** | "
          f"{avg(nm,rare,1):.1f} | {avg(nm,oidx,0):.1f} | {avg(nm,oidx,0)/len(oidx)*100:.1f}% |")
    W("")
    b2, f2, n2 = avg("기준(112)", tgt, 1), avg("전체블록(122)", tgt, 1), avg("비관계형만(117)", tgt, 1)
    ba, fa = avg("기준(112)", oidx, 0) / len(oidx) * 100, avg("전체블록(122)", oidx, 0) / len(oidx) * 100
    ok2, ok3, ok4 = f2 > b2, (fa - ba) >= -2.0, f2 > n2
    W("## 2) 판정")
    W("")
    W("| 규칙 | 실측 | 충족 |")
    W("|---|---|---|")
    W(f"| ② 표적 top-2 > 기준 | {b2:.1f} → {f2:.1f} ({f2-b2:+.1f}) | {'○' if ok2 else '**✗**'} |")
    W(f"| ③ 전체 top-1 −2.0%p 이내 | {ba:.1f}% → {fa:.1f}% ({fa-ba:+.1f}%p) | {'○' if ok3 else '**✗**'} |")
    W(f"| ④ 관계형 > 비관계형 | 비관계형 {n2:.1f} · 전체 {f2:.1f} | {'○' if ok4 else '**✗**'} |")
    W("")
    W(f"**판정 → {'통과 — GNN 제작 근거 성립' if (ok2 and ok3 and ok4) else '미달 — GNN 축 닫음'}**")
    W("")
    W("## 3) 표적 클래스별 (시드평균 top-2)")
    W("")
    W("| 클래스 | 학습행 | 기준 | 전체블록 | 비관계형만 |")
    W("|---|---|---|---|---|")
    for c in sorted({y[r] for r in tgt}):
        ii = [r for r in tgt if y[r] == c]
        W(f"| {c} | {len(ii)} | {avg('기준(112)',ii,1):.1f} | {avg('전체블록(122)',ii,1):.1f} | "
          f"{avg('비관계형만(117)',ii,1):.1f} |")
    W("")
    W(f"## 4) 실측 · 총 {time.time()-t0:.0f}s")
    W("")
    W("사전 기대가 낮았던 이유(머리말): 3.5 AChA 는 정의 곁가지가 그래프에 없고(L-3.5 정의 분기점 0/4행), "
      "1.9·1.3 은 학습행이 1~2개다. 블록은 **분기점의 23%가 현재 피처 밖**이라는 근거로 돌렸다.")
    open(f"{R}/experiments/V1_vessel_axis/RESULTS_G0.md", "w").write("\n".join(O) + "\n")
    print("\n".join(O), flush=True)


if __name__ == "__main__":
    main()
