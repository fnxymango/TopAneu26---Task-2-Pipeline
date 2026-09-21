#!/usr/bin/env python3
"""V2-A 판정 — ICA 를 C6/C7/terminus 로 나눈 GT 혈관을 줬을 때 분류기가 얼마나 더 맞히나 (상한).

두 팔은 **혈관 입력만** 다르다. 둘 다 학습표·평가 모두 GT 혈관(위치·overlap 전부)에서 뽑는다.
  v2a36_pf  GT 혈관 36클래스 (ICA-C6-C7 한 덩어리)
  v2a40_pf  GT 혈관 40클래스 (v2_rule.json 규칙으로 분할)
검출기 b1ff · gC ON · 패치필터 · 개정판 GT 라벨 · 5시드 — 나머지 전부 동일.
⚠ 이건 **진단용 상한**이다. 평가에 GT 혈관을 쓰므로 제출 후보가 아니고 PROJECT_RULES.md 0-2 기준선을 옮기지 않는다.

── 판정규칙 (결과 보기 전에 고정 · 2026-09-14) ──────────────────────────────
주 지표 = **ICA 원위 클래스(3.2~3.7) 적중률** — 케이스의 GT 클래스가 예측에 등장하면 적중
          (공식 eval 의 존재기반 TP 와 같은 규약), test+val · 5시드 합산.
  Δ(40 − 36) ≥ +10%p  ∧  test·val 각각 Δ ≥ 0   → **축 실재.** V2-B(예측 혈관에 규칙 적용)로 간다
  Δ < +3%p                                       → **분할 정보가 분류기에 안 먹힌다.** ICA 분할 축 종결
  그 사이                                         → 보고하고 사용자 판단
안전 확인(판정 아님): 비ICA 클래스 적중 합계가 3% 넘게 줄면 명시한다.
참고로 신 eval 7지표(v2a40 − v2a36)도 붙인다.
───────────────────────────────────────────────────────────────────────────
"""
import json, os, re, subprocess, collections
import numpy as np, nibabel as nib

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
H = f"{R}/experiments/H1_patchfilter"; D = f"{R}/experiments/D1_newdata"
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
LOC = {int(k): v for k, v in S["location_classes"].items()}
DIST = {k for k, v in LOC.items() if re.match(r"^(?:[RL]-)?3\.[2-7]\s", v)}
ICA = {k for k, v in LOC.items() if re.match(r"^(?:[RL]-)?3\.\d\s", v)}
PY = os.path.expanduser("~/miniconda3/envs/sbaneu2/bin/python")


def tally(tag):
    out = collections.defaultdict(lambda: [0, 0])      # (split, group) -> [hit, tot]
    for sp in ("test", "val"):
        for c in S["splits"][sp]:
            gp = f"{R}/dataset/TopAneu/location_masks/{c}.nii.gz"
            if not os.path.exists(gp):
                continue
            cls = {int(x) for x in np.unique(np.asanyarray(nib.load(gp).dataobj)) if x}
            for sd in range(5):
                pp = f"{H}/pred/{tag}_{sp}_s{sd}/{c}.nii.gz"
                if not os.path.exists(pp):
                    continue
                pc = {int(x) for x in np.unique(np.asanyarray(nib.load(pp).dataobj)) if x}
                for k in cls:
                    g = "dist" if k in DIST else ("ica31" if k in ICA else "other")
                    out[(sp, g)][0] += k in pc; out[(sp, g)][1] += 1
    return out


def main():
    rule = json.load(open(f"{D}/v2_rule.json"))
    print("# V2-A — ICA C6/C7/terminus 분할의 상한 (GT 혈관)\n")
    print(f"분할 규칙 **{rule['name']}** · 파라미터 {json.dumps(rule['params'], ensure_ascii=False)}\n")
    a36, a40 = tally("v2a36_pf"), tally("v2a40_pf")
    print("## 1. ICA 원위(3.2~3.7) 적중 — 주 지표\n")
    print("| split | 36클래스 | 40클래스 | Δ |")
    print("|---|---|---|---|")
    rates = {}
    for sp in ("test", "val", "합계"):
        keys = [("test",), ("val",)] if sp == "합계" else [(sp,)]
        h36 = sum(a36[(k[0], "dist")][0] for k in keys); n36 = sum(a36[(k[0], "dist")][1] for k in keys)
        h40 = sum(a40[(k[0], "dist")][0] for k in keys); n40 = sum(a40[(k[0], "dist")][1] for k in keys)
        r36, r40 = h36 / max(n36, 1), h40 / max(n40, 1)
        rates[sp] = r40 - r36
        print(f"| {sp} | {h36}/{n36} = {r36:.1%} | {h40}/{n40} = {r40:.1%} | **{(r40-r36)*100:+.1f}%p** |")
    print("\n## 2. 나머지 클래스 (안전 확인)\n")
    print("| 그룹 | 36클래스 | 40클래스 | Δ |")
    print("|---|---|---|---|")
    for g, nm in (("ica31", "ICA 3.1 C1-C5"), ("other", "비ICA")):
        h36 = sum(a36[(sp, g)][0] for sp in ("test", "val")); n = sum(a36[(sp, g)][1] for sp in ("test", "val"))
        h40 = sum(a40[(sp, g)][0] for sp in ("test", "val"))
        print(f"| {nm} | {h36}/{n} | {h40}/{n} | {h40-h36:+d} |")
    oth36 = sum(a36[(sp, "other")][0] for sp in ("test", "val"))
    oth40 = sum(a40[(sp, "other")][0] for sp in ("test", "val"))
    harm = oth36 > 0 and (oth36 - oth40) / oth36 > 0.03
    print("\n## 3. 판정\n")
    d = rates["합계"]
    if d >= 0.10 and rates["test"] >= 0 and rates["val"] >= 0:
        v = "축 실재 — V2-B(예측 혈관에 규칙 적용)로 간다"
    elif d < 0.03:
        v = "분할 정보가 분류기에 안 먹힌다 — ICA 분할 축 종결"
    else:
        v = "중간 — 보고하고 사용자 판단"
    print(f"ICA 원위 적중 Δ 합계 {d*100:+.1f}%p · test {rates['test']*100:+.1f}%p · val {rates['val']*100:+.1f}%p")
    print(f"\n**→ {v}**")
    if harm:
        print(f"\n⚠ 비ICA 적중이 {oth36}→{oth40} 로 3% 넘게 줄었다.")
    print("\n## 4. 참고 · 신 eval 7지표 (v2a40_pf − v2a36_pf)\n")
    print(subprocess.run([PY, f"{D}/g2_seeds.py", f"{H}/scores", "v2a40_pf", "v2a36_pf",
                          "GT혈관 40클래스 − 36클래스"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
