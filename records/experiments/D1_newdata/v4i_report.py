#!/usr/bin/env python3
"""V4-I 판정 보고서 — 규칙은 V4I.sh 머리말에 결과 보기 전에 고정."""
import json, os, glob, subprocess, collections, re
import numpy as np, nibabel as nib

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
H = f"{R}/experiments/H1_patchfilter"; D = f"{R}/experiments/D1_newdata"
PY = os.path.expanduser("~/miniconda3/envs/sbaneu2/bin/python")
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
LOC = {int(k): v for k, v in S["location_classes"].items()}


def run(*a):
    return subprocess.run([PY, *a], capture_output=True, text=True).stdout


print("# V4-I — ICA 원위 기하 룰 (예측 혈관 조건 재보정)\n")
print("판정규칙은 V4I.sh 머리말에 결과 보기 전에 고정. 1차=병변 TP · 안전=새로 틀린 ≤ 새로 맞힌 · 2차=ΔMCC ≥ −0.005\n")

# ── 0. 룰이 실제로 무엇을 건드렸나 ─────────────────────────────────────────
print("## 0. 룰 발동 내역 (패치필터 전 · c5 단계)\n")
safe = {}
for sp in ("test", "val"):
    rows = []
    for f in sorted(glob.glob(f"{D}/v4i_dump_{sp}_s*.jsonl")):
        sd = re.search(r"_s(\d)\.jsonl", f).group(1)
        rows += [dict(json.loads(l), seed=sd) for l in open(f)]
    fixed = sum(LOC.get(r["gtid"]) == r["new"] for r in rows)
    broke = sum(LOC.get(r["gtid"]) == r["old"] for r in rows)
    neutral = len(rows) - fixed - broke
    safe[sp] = (fixed, broke)
    print(f"**{sp}** 발동 {len(rows)}건 · 새로 맞힘 {fixed} · 새로 틀림 {broke} · 둘 다 아님 {neutral}")
    if rows:
        print("\n| 시드 | 케이스 | 병변 | RF 1등 → 룰 | GT | 결과 |\n|---|---|---|---|---|---|")
        for r in rows:
            g = LOC.get(r["gtid"], "GT 없음(오탐)")
            res = "✓ 수정" if g == r["new"] else ("✗ 파손" if g == r["old"] else "–")
            print(f"| s{r['seed']} | {r['case']} | {r['les']} | {r['old']} → {r['new']} | {g} | {res} |")
    print()

# ── 1. 룰 미발동 케이스가 기준선과 같은가 (구현 검증) ────────────────────────
print("## 1. 구현 검증 — 룰이 안 걸린 케이스는 기준선과 같아야 한다\n")
touched = collections.defaultdict(set)
for f in glob.glob(f"{D}/v4i_dump_*.jsonl"):
    m = re.search(r"v4i_dump_(\w+)_s(\d)\.jsonl", f)
    for l in open(f):
        touched[(m.group(1), m.group(2))].add(json.loads(l)["case"])
same = diff = 0; bad = []
for sp in ("test", "val"):
    for sd in range(5):
        for p in sorted(glob.glob(f"{H}/pred/b1v4i_{sp}_s{sd}/*.nii.gz")):
            c = os.path.basename(p)[:-7]
            if c in touched[(sp, str(sd))]:
                continue
            q = f"{H}/pred/b1Non_{sp}_s{sd}/{c}.nii.gz"
            if not os.path.exists(q):
                continue
            a = np.asanyarray(nib.load(p).dataobj); b = np.asanyarray(nib.load(q).dataobj)
            if a.shape == b.shape and (a == b).all():
                same += 1
            else:
                diff += 1; bad.append(f"{sp}_s{sd}/{c}")
print(f"미발동 케이스 동일 {same} · 다름 {diff}" + (f" ★ {bad[:5]}" if diff else " ✓"))

# ── 2. 1차 · 병변 단위 TP ──────────────────────────────────────────────────
print("\n## 2. 1차 · 병변 단위 TP (b1Non_pf → b1v4i_pf)\n")
t1 = run(f"{D}/tpcount.py", "b1Non_pf", "b1v4i_pf")
print(t1)
m = re.search(r"Δ ([+-]?\d+)", t1)
d_tp = int(m.group(1)) if m else 0

# ── 3. 2차 · 7지표 ─────────────────────────────────────────────────────────
print("## 3. 2차 · 신 eval 7지표 (b1v4i_pf − b1Non_pf)\n")
t2 = run(f"{D}/g2_seeds.py", f"{H}/scores", "b1v4i_pf", "b1Non_pf", "ICA 원위 룰 − 기준선")
print(t2)
dm = dict(re.findall(r"## (test|val) · 시드.*?평균ΔMCC ([+-]?\d+\.\d+)", t2, re.S))

# ── 4. 판정 ────────────────────────────────────────────────────────────────
print("## 4. 판정\n")
ok1 = d_tp > 0
ok_s = all(safe[sp][1] <= safe[sp][0] for sp in ("test", "val"))
ok2 = all(float(dm.get(sp, "-1")) >= -0.005 for sp in ("test", "val"))
print(f"- 1차 병변 TP Δ {d_tp:+d} → {'충족' if ok1 else '미충족'}")
print(f"- 안전 test 수정 {safe['test'][0]}/파손 {safe['test'][1]} · val 수정 {safe['val'][0]}/파손 {safe['val'][1]} → {'충족' if ok_s else '위반'}")
print(f"- 2차 ΔMCC test {dm.get('test')} · val {dm.get('val')} → {'충족' if ok2 else '미충족'}")
print(f"\n**→ {'채택' if (ok1 and ok_s and ok2) else '기각' if (not ok1 or not ok_s) else '미채택'}**")
print("\n⚠ 룰은 ICA 원위 병변만 건드린다. τ3.0 은 train 격자 {0.5,1,2,3} 의 끝값이다(격자는 결과 전 고정).")
