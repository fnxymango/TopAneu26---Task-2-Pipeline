#!/usr/bin/env python3
"""T1 — 여러 분절에 걸친 병변을 GT 가 어떤 규약으로 라벨하는가 (train 만 · test·val 안 봄).

유형(type_masks) 을 스위치로 쓰기 위한 선행 측정. 방추형 OOF 정확도 53% 의 오답이 '몸통 ↔ 접합' 에 몰렸다.
병변마다 GT 혈관 라벨 구성(병변 복셀 + 1mm 외곽, 좌우 접두 제거한 혈관명)을 재고, 두 혈관 이상에 걸친 병변(각 ≥15%)에서
GT 위치 클래스가 아래 규약 중 어느 것과 맞는지 센다.
  규약 M (최다 혈관 몸통) : 가장 많이 덮은 혈관의 몸통 클래스
  규약 P (근위 혈관 몸통) : 걸친 혈관 중 해부학적으로 가장 근위인 혈관의 몸통 클래스
  규약 J (접합)          : 걸친 두 혈관이 만드는 접합 클래스
  몸통/접합 대응은 JUNC·TRUNK 표(해부학 정의 · 결과 보기 전 작성). 표에 없는 조합은 '정의 없음' 으로 따로 센다.

── 판정 (결과 보기 전 고정 · 2026-09-15) ────────────────────────────────────
 비낭형 걸침 병변에서 어느 한 규약이 ≥ 80% 설명 ∧ 그 병변 ≥ 10개 → T1 규칙 후보(e2e 설계로)
 아니면 T1·T4 를 닫는다. 낭형 걸침 병변도 같은 표로 적어 대조한다.
출력: t1_convention.json · 표준출력(RESULTS_T1.md 로 복사)
"""
import json, os, re, collections
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
ST = np.ones((3, 3, 3), bool)
ORDER = ["VA", "PICA", "BA", "AICA", "SCA", "P1P2", "P3P4", "ICA-C1-C5", "ICA-C6-C7", "OA", "Pcom", "AChA",
         "M1", "M2", "M3", "A1A2", "Acom", "A3", "3rd-A2", "3rd-A3"]          # 근위 → 원위 (계통별)
TRUNK = {"VA": "1.1", "PICA": "1.2", "BA": "1.4", "AICA": "1.6", "SCA": "1.8", "P1P2": "2.1", "P3P4": "2.2",
         "ICA-C1-C5": "3.1", "ICA-C6-C7": "3.6", "M1": "5.1", "M2": "5.3d", "M3": "5.3d", "A1A2": "4.2", "A3": "4.4"}
JUNC = {frozenset(("VA", "PICA")): "1.3", frozenset(("VA", "BA")): "1.5", frozenset(("BA", "AICA")): "1.7",
        frozenset(("BA", "SCA")): "1.9", frozenset(("BA", "P1P2")): "1.10", frozenset(("P1P2", "P3P4")): "2.1",
        frozenset(("ICA-C1-C5", "ICA-C6-C7")): "3.3", frozenset(("ICA-C6-C7", "OA")): "3.2",
        frozenset(("ICA-C6-C7", "Pcom")): "3.4", frozenset(("ICA-C6-C7", "AChA")): "3.5",
        frozenset(("ICA-C6-C7", "M1")): "3.7", frozenset(("ICA-C6-C7", "A1A2")): "3.7",
        frozenset(("M1", "M2")): "5.3j", frozenset(("A1A2", "Acom")): "4.1", frozenset(("A1A2", "A3")): "4.3"}


def code(nm):
    b = re.sub(r"^[RL]-", "", nm); c = b.split()[0]
    return ("5.3j" if "M1-M2" in b else "5.3d") if c == "5.3" else c


def case(cid, rows):
    import sys
    sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); os.environ.setdefault("TOPANEU_ROOT", R)
    import d9xx_lib as L
    names = L.vessel_dense_names()
    li = nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz")
    loc = np.asanyarray(li.dataobj); sp = np.array(li.header.get_zooms()[:3], float)
    ves = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz").dataobj)
    ty = json.load(open(f"{D}/k6_type_feat.json"))
    lab, _ = ndimage.label(loc > 0, structure=ST)
    it = max(1, int(round(1.0 / sp.min())))
    out = []
    for r in rows:
        m = lab == r["lesion_mask_idx"]
        idx = np.argwhere(m); lo = np.maximum(idx.min(0) - it - 1, 0); hi = np.minimum(idx.max(0) + it + 2, loc.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        reg = ndimage.binary_dilation(m[sl], structure=ST, iterations=it)
        v = ves[sl][reg]; v = v[v > 0]
        comp = collections.Counter(re.sub(r"^[RL]-", "", names[int(x)]) for x in v)
        tot = sum(comp.values())
        frac = {k: n / tot for k, n in comp.items()} if tot else {}
        out.append(dict(key=f"{cid}|{r['lesion_mask_idx']}", gt=code(r["gt_loc"]), frac=frac,
                        nonsac=int(ty[f"{cid}|{r['lesion_mask_idx']}"]["type"] in (2, 3))))
    return out


def conventions(frac):
    span = sorted([k for k, f in frac.items() if f >= 0.15], key=lambda k: -frac[k])
    if len(span) < 2:
        return None
    major = span[0]
    prox = min(span, key=lambda k: ORDER.index(k) if k in ORDER else 99)
    j = JUNC.get(frozenset(span[:2]))
    return dict(span=span, M=TRUNK.get(major), P=TRUNK.get(prox), J=j)


def main():
    import multiprocessing as mp
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    res = []
    with mp.Pool(6) as p:
        for o in p.starmap(case, list(by.items()), chunksize=1):
            res += o
    json.dump(res, open(f"{D}/t1_convention.json", "w"))
    print("# T1 — 걸친 병변의 GT 라벨 규약 (train)\n")
    verdict = {}
    for grp, name in ((1, "비낭형"), (0, "낭형")):
        sub = [x for x in res if x["nonsac"] == grp]
        sp = [(x, conventions(x["frac"])) for x in sub]
        sp = [(x, c) for x, c in sp if c]
        cnt = collections.Counter()
        for x, c in sp:
            for k in ("M", "P", "J"):
                cnt[k] += c[k] == x["gt"]
            cnt["없음"] += not any(c[k] == x["gt"] for k in ("M", "P", "J"))
        n = len(sp)
        print(f"## {name} — 병변 {len(sub)} · 두 혈관 이상 걸침 {n}\n")
        if n:
            print(f"규약 M(최다 몸통) {cnt['M']}/{n} ({cnt['M']/n:.0%}) · P(근위 몸통) {cnt['P']}/{n} ({cnt['P']/n:.0%}) · "
                  f"J(접합) {cnt['J']}/{n} ({cnt['J']/n:.0%}) · 어느 것도 아님 {cnt['없음']}\n")
            print("| GT | 걸친 혈관(비율) | M | P | J |\n|---|---|---|---|---|")
            for x, c in sorted(sp, key=lambda t: t[0]["gt"]):
                fr = " · ".join(f"{k} {x['frac'][k]:.2f}" for k in c["span"])
                mk = lambda k: "✓" if c[k] == x["gt"] else (c[k] or "–")
                print(f"| {x['gt']} | {fr} | {mk('M')} | {mk('P')} | {mk('J')} |")
            print()
        if grp == 1:
            best = max(("M", "P", "J"), key=lambda k: cnt[k]) if n else None
            ok = bool(n >= 10 and best and cnt[best] / n >= 0.80)
            verdict = dict(n=n, best=best, rate=(cnt[best] / n if n else 0), ok=ok)
    print(f"**판정: 비낭형 걸침 {verdict['n']}개 · 최선 규약 {verdict['best']} {verdict['rate']:.0%} → "
          f"{'규칙 후보 (≥80% ∧ ≥10개) · e2e 설계로' if verdict['ok'] else '미달 · T1·T4 닫음'}**")
    json.dump(verdict, open(f"{D}/t1_verdict.json", "w"))


if __name__ == "__main__":
    main()
