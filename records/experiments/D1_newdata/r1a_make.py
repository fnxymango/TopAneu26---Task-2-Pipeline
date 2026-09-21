#!/usr/bin/env python3
"""R1-A — '그 환자에게 없는 곁가지의 접합 클래스는 찍지 않는다' 규칙을 추론에 넣은 c5 복사본(c5_r1a.py) 생성.

train OOF 스크리닝(r1_rules.py · RESULTS_R1.md): 바꾼 30건 중 살아남 11 · 새로 틀림 2 (5.5:1),
사전 고정 관문의 양 조건(시드당 ≥5)에는 미달했다. 판정은 e2e K0 장치로 직접 받는다(사용자 지시 2026-09-16).

구현: 케이스의 **예측 혈관 마스크**에서 곁가지 라벨 복셀이 TOPANEU_R1A_MINVOX(기본 20) 미만이면
그 곁가지가 필요한 접합 클래스의 확률을 0 으로 만든다. _proba 한 곳만 고치므로 1등 선택·gC 2등 조각·β 보정이
전부 같은 제약 위에서 동작한다(1등만 바꾸면 gC 가 같은 클래스를 다시 배출해 규칙이 반쯤 무효가 된다).
대상: 3.2(OA) · 3.4(Pcom) · 3.5(AChA) · 1.3(PICA) · 1.7(AICA) · 1.9(SCA) · 4.1(Acom)
TOPANEU_R1A=0(기본) 이면 원본과 완전히 같다.
"""
import hashlib
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; S = f"{R}/code/sblee/nnunet/scripts"
s = open(f"{S}/c5_location_v2.py").read()
md5 = hashlib.md5(s.encode()).hexdigest()

old1 = '''def _proba(model, r):'''
new1 = '''# ── R1-A · 없는 곁가지의 접합 클래스 금지 (experiments/D1_newdata/r1a_make.py 가 삽입) ──
R1A = int(os.environ.get("TOPANEU_R1A", "0"))
R1A_MINVOX = int(os.environ.get("TOPANEU_R1A_MINVOX", "20"))
R1A_BRANCH = {"3.2": "OA", "3.4": "Pcom", "3.5": "AChA", "1.3": "PICA", "1.7": "AICA", "1.9": "SCA", "4.1": "Acom"}
_R1A_BAN = set()          # 이번 케이스에서 금지된 위치클래스 이름들


def r1a_set_case(ves, ves_names, id2name=None):
    """예측 혈관에서 복셀이 거의 없는 곁가지를 찾아, 그 곁가지가 필요한 접합 클래스를 금지 집합에 넣는다."""
    global _R1A_BAN
    _R1A_BAN = set()
    if not R1A:
        return
    cnt = np.bincount(np.asarray(ves).ravel().astype(np.int64))
    have = {ves_names[i] for i in ves_names if i < len(cnt) and cnt[i] >= R1A_MINVOX}
    names = id2name.values() if id2name else _R1A_ALLNAMES
    for nm in names:
        b = re.sub(r"^[RL]-", "", str(nm)).split()[0]
        br = R1A_BRANCH.get(b)
        if not br:
            continue
        sd = str(nm)[0] if str(nm)[:2] in ("R-", "L-") else None
        want = f"{sd}-{br}" if (sd and br != "Acom") else br
        if want not in have:
            _R1A_BAN.add(str(nm))


def _proba(model, r):'''

old2 = '''    p = model["clf"].predict_proba(v[None, :])[0]
    if USE_TTA:'''
new2 = '''    p = model["clf"].predict_proba(v[None, :])[0]
    if R1A and _R1A_BAN:
        p = p.copy()
        for _j, _c in enumerate(cls):
            if str(_c) in _R1A_BAN:
                p[_j] = 0.0
        _s = p.sum()
        if _s > 0:
            p = p / _s
    if USE_TTA:'''

old3 = '''        out = np.zeros(gt.shape, dtype=np.int32)
        for r in rows:'''
new3 = '''        if R1A:
            r1a_set_case(ves, ves_names, id2name)
        out = np.zeros(gt.shape, dtype=np.int32)
        for r in rows:'''
for o in (old1, old2, old3):
    assert s.count(o) == 1, (o[:40], s.count(o))
s = s.replace(old1, new1).replace(old2, new2).replace(old3, new3)
s = s.replace("import argparse, re, json, math, os, sys, time, collections",
              "import argparse, re, json, math, os, sys, time, collections\n_R1A_ALLNAMES = []", 1)
# 전체 클래스 이름표(금지 판정 대상)는 official_location_names 로 채운다
s = s.replace("    id2name, name2id = L.official_location_names()",
              "    id2name, name2id = L.official_location_names()\n    globals()['_R1A_ALLNAMES'] = list(id2name.values())")
open(f"{S}/c5_r1a.py", "w").write(f"# 자동생성: experiments/D1_newdata/r1a_make.py · 원본 c5_location_v2.py md5 {md5}\n" + s)
print("c5_r1a.py 생성")
