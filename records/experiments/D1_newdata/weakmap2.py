#!/usr/bin/env python3
"""제출본 기준 분류 약점 — 4영역을 14개 세부구간으로 쪼갠 표."""
import json, re, collections
D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
L = json.load(open(f"{D}/intweak_b1on_pf.json"))["lesions"]
det = [x for x in L if x["detected"]]
TOT = sum(x["n_seed"] for x in det); OK = sum(x["n_ok"] for x in det)

# 병변 클래스 → 세부구간 (좌우 접두 제거한 번호 기준)
SUB = [
    ("후순환", "VA · PICA",        ["1.1", "1.2", "1.3"]),
    ("후순환", "BA 몸통 · 분지",   ["1.4", "1.5", "1.6", "1.7", "1.8", "1.9"]),
    ("후순환", "BA tip",           ["1.10"]),
    ("후순환", "PCA",              ["2.1", "2.2"]),
    ("ICA",    "C1-C5 (해면동 아래)", ["3.1"]),
    ("ICA",    "C6 (안동맥 구간)",  ["3.2", "3.3"]),
    ("ICA",    "C7 (교통동맥 구간)", ["3.4", "3.5", "3.6"]),
    ("ICA",    "C7 종말부",        ["3.7"]),
    ("ACA",    "Acom",             ["4.1"]),
    ("ACA",    "A1",               ["4.2"]),
    ("ACA",    "A2 이원위",        ["4.3", "4.4", "4.5"]),
    ("MCA",    "M1 몸통",          ["5.1"]),
    ("MCA",    "M1 분기부",        ["5.2", "5.3 M1-M2"]),
    ("MCA",    "M2/M3 원위",       ["5.3 Distal"]),
]
def num(name):
    n = re.sub(r"^[RL]-", "", name)
    if n.startswith("5.3"):
        return "5.3 Distal" if "Distal" in n else "5.3 M1-M2"
    return re.match(r"([\d.]+?)\s", n).group(1)

key = {}
for terr, sub, ns in SUB:
    for n in ns:
        key[n] = (terr, sub)

agg = collections.defaultdict(lambda: [0, 0, 0])          # 병변, 시드판정, 정답
bad = collections.defaultdict(collections.Counter)        # 전시드 오답 내역
miss = []
for x in det:
    k = key.get(num(x["name"]))
    if k is None:
        miss.append(x["name"]); continue
    a = agg[k]; a[0] += 1; a[1] += x["n_seed"]; a[2] += x["n_ok"]
    if x["n_ok"] == 0:
        bad[k][x["name"]] += 1
assert not miss, miss

print(f"검출된 GT 병변 {len(det)}개 · 시드판정 {TOT}회 · 오답 {TOT-OK} ({(TOT-OK)/TOT:.1%})\n")
print("| 영역 | 세부구간 | 병변 | 시드판정 | 정답률 | 오답 | 전체 오답 중 |")
print("|---|---|---|---|---|---|---|")
rows = [(t, s, *agg[(t, s)]) for t, s, _ in SUB if agg[(t, s)][1]]
for t, s, n, tt, ok in sorted(rows, key=lambda r: (r[4] / r[3])):
    print(f"| {t} | {s} | {n} | {tt} | **{ok/tt:.1%}** | {tt-ok} | {(tt-ok)/(TOT-OK):.0%} |")

print("\n## 전시드 오답(편향)이 몰린 세부구간\n")
print("| 영역 | 세부구간 | 전시드 오답 병변 |")
print("|---|---|---|")
for t, s, _ in SUB:
    b = bad.get((t, s))
    if b:
        print(f"| {t} | {s} | {', '.join(f'{k}×{v}' if v>1 else k for k,v in b.most_common())} |")
