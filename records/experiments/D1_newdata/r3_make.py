#!/usr/bin/env python3
"""R3 — 접합 클래스 거리 상한 규칙을 추론에 넣은 c5 복사본(c5_r3.py) 생성.

규칙: 1등 후보가 접합 클래스인데, 그 접합이 요구하는 **분기점까지의 거리(추론 조건 bp_mm)** 가 τ 초과(또는 노드 없음)면
그 클래스의 확률을 0 으로 만든다. _proba 한 곳만 고쳐 1등 선택·gC 2등 조각·β 보정이 모두 같은 제약 위에서 돈다.
τ 는 train OOF 스크리닝(r3_dist.py · RESULTS_R3_SCREEN.md)에서 4mm 로 정했다 — test·val 은 쓰지 않았다.
TOPANEU_R3=0(기본)이면 원본과 완전히 같다.
"""
import hashlib
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; S = f"{R}/code/sblee/nnunet/scripts"
s = open(f"{S}/c5_location_v2.py").read()
md5 = hashlib.md5(s.encode()).hexdigest()

old1 = "def _proba(model, r):"
new1 = '''# ── R3 · 접합 클래스 거리 상한 (experiments/D1_newdata/r3_make.py 가 삽입) ──
R3 = int(os.environ.get("TOPANEU_R3", "0"))
R3_TAU = float(os.environ.get("TOPANEU_R3_TAU", "4.0"))
R3_PAIR = {"3.2": ("ICA-C6-C7", "OA"), "3.4": ("ICA-C6-C7", "Pcom"), "3.5": ("ICA-C6-C7", "AChA"),
           "1.3": ("VA", "PICA"), "1.7": ("BA", "AICA"), "1.9": ("BA", "SCA"), "4.1": ("A1A2", "Acom")}
_R3_IDX = None            # 위치클래스 이름 → bp_mm 인덱스


def _r3_index(cls_names):
    global _R3_IDX
    if _R3_IDX is not None:
        return _R3_IDX
    pidx = {frozenset(p): i for i, p in enumerate(JUNCTION_PAIRS)}
    _R3_IDX = {}
    for nm in cls_names:
        nm = str(nm)
        c = re.sub(r"^[RL]-", "", nm).split()[0]
        pr = R3_PAIR.get(c)
        if pr is None:
            continue
        sd = nm[0] if nm[:2] in ("R-", "L-") else None
        f = lambda v: v if v == "Acom" else (f"{sd}-{v}" if sd else v)
        k = pidx.get(frozenset((f(pr[0]), f(pr[1]))))
        if k is None and sd is None:      # 좌우 없는 클래스(4.1 등)는 양쪽 다 본다
            ks = [pidx.get(frozenset((f"{x}-{pr[0]}" if pr[0] != "Acom" else pr[0],
                                      f"{x}-{pr[1]}" if pr[1] != "Acom" else pr[1]))) for x in ("R", "L")]
            ks = [x for x in ks if x is not None]
            if ks:
                _R3_IDX[nm] = ks
            continue
        if k is not None:
            _R3_IDX[nm] = [k]
    return _R3_IDX


def _r3_ban(cls, r):
    """이 병변에서 금지할 클래스 인덱스 집합 (분기점이 τ 밖이거나 없음)."""
    idx = _r3_index(cls)
    bp = r.get("bp_mm") or []
    ban = set()
    for j, c in enumerate(cls):
        ks = idx.get(str(c))
        if not ks:
            continue
        ds = []
        for k in ks:
            v = bp[k] if k < len(bp) else None
            ds.append(float("inf") if v is None else float(v))
        if min(ds) > R3_TAU:
            ban.add(j)
    return ban


def _proba(model, r):'''

old2 = '''    p = model["clf"].predict_proba(v[None, :])[0]
    if USE_TTA:'''
new2 = '''    p = model["clf"].predict_proba(v[None, :])[0]
    if R3:
        _ban = _r3_ban(cls, r)
        if _ban:
            p = p.copy()
            for _j in _ban:
                p[_j] = 0.0
            _s = p.sum()
            if _s > 0:
                p = p / _s
    if USE_TTA:'''
for o in (old1, old2):
    assert s.count(o) == 1, (o[:30], s.count(o))
open(f"{S}/c5_r3.py", "w").write(f"# 자동생성: experiments/D1_newdata/r3_make.py · 원본 c5_location_v2.py md5 {md5}\n"
                                 + s.replace(old1, new1).replace(old2, new2))
print("c5_r3.py 생성")
