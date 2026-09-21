#!/usr/bin/env python3
"""R1 — 애매한 병변에만 거는 후처리 규칙 2종 스크리닝 (train OOF 만 · test·val 안 봄).

분류기 내부는 건드리지 않는다. 이미 나온 1등/2등 위에 **케이스 해부 정보**로 덮어쓰는 규칙이다.

규칙 A · 해부 제약 — 그 환자에게 없는 곁가지의 접합 클래스는 찍지 않는다
  예측 혈관(vespp_train)에 곁가지 라벨 복셀이 MIN_VOX 미만이면 그 곁가지 접합 클래스를 1등에서 빼고 2등으로 내린다.
  대상: 3.2(OA) · 3.4(Pcom) · 3.5(AChA) · 1.3(PICA) · 1.7(AICA) · 1.9(SCA) · 4.1(Acom)
  근거: 앵커가 아니라 **곁가지 자체가 없는** 경우가 실제로 많다(C60 §34 · B1 관문 V 에서 참조 그래프에 없는 단위 다수).

규칙 B · 유형 스위치 — 비낭형이면 후순환 접합 대신 몸통
  유형 예측기(K6-1 · OOF AUC 0.924)가 비낭형(방추·박리)이라고 하면, 후순환 접합 1등을 몸통으로 바꾼다.
  1.3→1.1 · 1.5→1.4 · 1.9→1.4 · 1.10→1.4 (좌우는 1등 것을 따른다)
  근거: train 비낭형 41개 중 34개가 후순환이고 대부분 VA·BA **몸통**(1.1 17/18 · 1.4 9/11).
  주의: 유형별 사전확률 전역 보정(9/15 빠른 확인 · 7고침/24망침)과 다르다 — 대상이 후순환 접합 4클래스로 좁다.

측정: 케이스 묶음 5폴드 × 시드 0~4 OOF. 규칙이 1등을 바꾼 병변만 세어 '살아남(오답→정답)'·'새로 틀림(정답→오답)'.
── 관문 (결과 보기 전 고정 · 2026-09-16) ─────────────────────────────────
 규칙별 통과 = [살아남 ≥ 2 × 새로 틀림] ∧ [살아남 ≥ 5 (5시드 평균)] → e2e 후보(K0 장치로 판정)
 둘 다 미달이면 그 규칙은 닫는다. 참고로 두 규칙을 함께 건 경우도 같이 적는다(판정은 개별 규칙으로).
"""
import json, os, sys, re, collections
import numpy as np, nibabel as nib

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D)
os.environ.setdefault("TOPANEU_ROOT", R)
MIN_VOX = 20
BRANCH = {"3.2": "OA", "3.4": "Pcom", "3.5": "AChA", "1.3": "PICA", "1.7": "AICA", "1.9": "SCA", "4.1": "Acom"}
TRUNK = {"1.3": "1.1", "1.5": "1.4", "1.9": "1.4", "1.10": "1.4"}


def code(nm):
    b = re.sub(r"^[RL]-", "", nm); return b.split()[0]


def side(nm):
    m = re.match(r"^([RL])-", nm); return m.group(1) if m else None


def branch_presence():
    """케이스별 (예측 혈관에서) 라벨 복셀 수 ≥ MIN_VOX 인 혈관 이름 집합."""
    import d9xx_lib as L
    names = L.vessel_dense_names()
    out = {}
    fp = f"{D}/r1_presence.json"
    if os.path.exists(fp):
        return {k: set(v) for k, v in json.load(open(fp)).items()}
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    for cid in sorted({r["case"] for r in rows}):
        v = np.asanyarray(nib.load(f"{R}/experiments/_c1_realpred/vespp_train/{cid}.nii.gz").dataobj)
        cnt = np.bincount(v.ravel().astype(np.int64))
        out[cid] = {names[i] for i in range(1, min(len(cnt), 37)) if cnt[i] >= MIN_VOX}
    json.dump({k: sorted(v) for k, v in out.items()}, open(fp, "w"))
    return out


def type_oof(rows, fold):
    """K6-1 형상 피처로 비낭형 확률 OOF (같은 폴드)."""
    from sklearn.ensemble import RandomForestClassifier
    T = json.load(open(f"{D}/k6_type_feat.json"))
    keys = [f"{r['case']}|{r['lesion_mask_idx']}" for r in rows]
    X = np.array([T[k]["x"] for k in keys]); y = np.array([T[k]["type"] in (2, 3) for k in keys], int)
    p = np.zeros(len(rows))
    for k in range(5):
        te = np.array([fold[r["case"]] == k for r in rows])
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=4)
        clf.fit(X[~te], y[~te]); p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def job(sd):
    os.environ["CLF_SEED"] = str(sd)
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    cases = sorted({r["case"] for r in rows})
    perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
    top = [None] * len(rows)
    for k in range(5):
        m = C5.fit_model([r for r in rows if fold[r["case"]] != k], ax, kind="rf", mirror=True)
        for i, r in enumerate(rows):
            if fold[r["case"]] == k:
                nm, pb, _ = C5.predict_ranked(m, r, 0.5)
                top[i] = [str(x) for x in nm[:3]] if nm is not None and len(nm) else []
    return sd, top, fold


def main():
    from multiprocessing import Pool
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    truth = [r["gt_loc"] for r in rows]
    pres = branch_presence()
    with Pool(5) as p:
        res = p.map(job, range(5))
    stat = collections.defaultdict(lambda: collections.Counter())
    for sd, top, fold in res:
        tp = type_oof(rows, fold)
        for i, r in enumerate(rows):
            t1 = top[i][0] if top[i] else None   # 피처가 전부 0 이면 기권 → 규칙 대상 아님
            if t1 is None:
                continue
            t2 = top[i][1] if len(top[i]) > 1 else None
            c1, sd_ = code(t1), side(t1)
            new = {}
            b = BRANCH.get(c1)
            if b and t2:
                nm = f"{sd_}-{b}" if sd_ and b != "Acom" else b
                if nm not in pres[r["case"]]:
                    new["A"] = t2
            if tp[i] >= 0.5 and c1 in TRUNK:
                new["B"] = TRUNK[c1]      # 몸통 코드 · 이름은 아래에서 공식 이름표로 찾는다
            for rule, val in new.items():
                nm_new = val
                if rule == "B":
                    nm_new = next((n for n in NAMES if code(n) == val and (side(n) == sd_ or side(n) is None)), None)
                    if nm_new is None:
                        continue
                if nm_new == t1:
                    continue
                ok_old, ok_new = t1 == truth[i], nm_new == truth[i]
                stat[rule]["바뀜"] += 1
                stat[rule]["살아남"] += (not ok_old) and ok_new
                stat[rule]["새로틀림"] += ok_old and (not ok_new)
    print("# R1 — 후처리 규칙 스크리닝 (train OOF · 시드 0~4 · 원본 병변 271)\n")
    print("| 규칙 | 1등이 바뀐 병변(5시드 합) | 살아남 | 새로 틀림 | 순증 | 시드평균 살아남 | 판정 |\n|---|---|---|---|---|---|---|")
    out = {}
    for rule, nm in (("A", "해부 제약(없는 곁가지의 접합 금지)"), ("B", "유형 스위치(비낭형 → 후순환 몸통)")):
        s = stat[rule]; fx, br = s["살아남"], s["새로틀림"]
        ok = (fx >= 2 * br) and (fx / 5 >= 5)
        out[rule] = dict(changed=s["바뀜"], fixed=fx, broken=br, ok=bool(ok))
        print(f"| {rule} {nm} | {s['바뀜']} | {fx} | {br} | {fx - br:+d} | {fx/5:.1f} | {'통과' if ok else '미달'} |")
    json.dump(out, open(f"{D}/r1_gate.json", "w"), indent=1)
    print("\n**관문(살아남 ≥ 2×새로틀림 ∧ 시드평균 살아남 ≥ 5) → " +
          (" · ".join(f"{k} {'통과' if v['ok'] else '미달'}" for k, v in out.items())) + "**")


if __name__ == "__main__":
    import d9xx_lib as L
    NAMES = list(json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["location_classes"].values())
    main()
