#!/usr/bin/env python3
"""V1-B — BA-SCA junction 가드 룰을 넣은 c5 복사본(c5_v1b.py)을 원본에서 기계적으로 만든다.

── 설계 근거 (학습표 train 271행 · 2026-09-14 실측) ─────────────────────────
PLAN.md V1-B 는 후순환 두 룰(1.9 BA-SCA · 1.3 VA-PICA)을 제안했지만 데이터가 절반만 지지했다.
  · VA-PICA: VA trunk 18개가 VA-PICA 분기점에 중앙 1.8mm 로 붙어 있고, VA-PICA junction 3개는
    오히려 7.2mm. 3개 중 2개는 PICA 혈관에 닿지도 않는다(99·7.8mm). 룰이 잡을 신호가 없다 → 제외.
  · VA-BA: BA 에 닿은 VA trunk 가 3/18 — 뒤집힘 위험이 이득과 비슷 → 제외.
  · **BA-SCA: 분리된다.** 기준은 분기점 거리가 아니라 **SCA 혈관 접촉**(dist_mm ≤ 0.5).
      1.9 BA-SCA junction 2개 → 같은 쪽 SCA 0.2 · 0.2 (둘 다 접촉)
      1.10 BA tip 8개      → 가장 가까운 SCA 2.1mm (접촉 0)
      1.4 BA trunk 11개    → 가장 가까운 SCA 0.8mm (접촉 0)
      2.1 P1P2 1개         → 1.6mm
    0.5mm 는 튜닝값이 아니라 '접촉'의 단위다 — VA trunk 병변이 자기 VA 에 닿으면 0.3~0.4 로 찍힌다.
  약점 지도(제출본 5시드): L-1.9 → 1.10 BA tip ×5, R-1.9 → 1.10 BA tip ×5 전시드 오답. 정확히 이 혼동이다.

── 룰 ────────────────────────────────────────────────────────────────────
  RF 1등이 BA 원위 이웃 {1.10 BA tip · 1.4 BA trunk · R/L-2.1 P1P2 · R/L-1.8 SCA trunk} 이고
  (1) 어느 쪽 SCA 혈관에 접촉(dist_mm ≤ 0.5)하고
  (2) 그 쪽 BA-SCA 분기점이 5mm 안에 있으면 (SCA 원위 병변 배제 — 분기부 '근방'의 넉넉한 정의)
  → 그 쪽 1.9 BA-SCA junction 으로 바꾼다. 양쪽 다 접촉하면 더 가까운 쪽.
  RF 확률 마진 가드는 **넣지 않는다.** 1.9 는 학습표본이 쪽마다 1개라 RF 가 1.10 을 확신하며 틀린다
  (전시드 오답 = 확신 오답). 마진 가드는 진짜 수정까지 막는다. 대신 기하 가드 두 개로 좁혔다.
  TOPANEU_V1B=0 이면 원본과 완전히 같다.
"""
import hashlib

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
S = f"{R}/code/sblee/nnunet/scripts"


def sub(src, old, new, what):
    assert src.count(old) == 1, f"{what}: 원문이 {src.count(old)}번"
    return src.replace(old, new)


s = open(f"{S}/c5_location_v2.py").read()
md5 = hashlib.md5(s.encode()).hexdigest()

s = sub(s, "def cmd_eval(args):", '''# ── V1-B · BA-SCA junction 가드 룰 (experiments/D1_newdata/v1b_make.py 가 삽입) ──────────
V1B_ON = os.environ.get("TOPANEU_V1B", "0") == "1"
V1B_TOUCH = float(os.environ.get("TOPANEU_V1B_TOUCH", "0.5"))   # 혈관 '접촉' 단위(mm)
V1B_NODE = float(os.environ.get("TOPANEU_V1B_NODE", "5.0"))     # BA-SCA 분기점 '근방'(mm)
V1B_DUMP = os.environ.get("TOPANEU_V1B_DUMP", "")
V1B_FROM = {"1.10 BA tip", "1.4 BA trunk", "R-2.1 P1P2", "L-2.1 P1P2",
            "R-1.8 SCA trunk", "L-1.8 SCA trunk"}
_V1B_NODE_IDX = {s_: next(i for i, p in enumerate(JUNCTION_PAIRS) if frozenset(p) == frozenset(("BA", f"{s_}-SCA")))
                 for s_ in ("R", "L")}


def v1b_override(name, r):
    """RF 1등 name 을 BA-SCA 룰로 바꿀지 결정. (새 이름, 발동 쪽 또는 None)"""
    if not V1B_ON or name not in V1B_FROM:
        return name, None
    dm = r.get("dist_mm") or {}
    bp = r.get("bp_mm") or []
    best = None
    for s_ in ("R", "L"):
        dv = dm.get(f"{s_}-SCA")
        if dv is None or dv > V1B_TOUCH:
            continue
        k = _V1B_NODE_IDX[s_]
        dn = bp[k] if k < len(bp) else None
        if dn is None or dn > V1B_NODE:
            continue
        if best is None or dv < best[0]:
            best = (dv, s_)
    if best is None:
        return name, None
    return f"{best[1]}-1.9 BA-SCA junction", best[1]


def cmd_eval(args):''', "v1b block")

s = sub(s, '''            name = predict_one_blend(mdl, r, args.beta, cp, args.crop_w)
            if DUMP_RANK:''', '''            name = predict_one_blend(mdl, r, args.beta, cp, args.crop_w)
            _v1b_old = name
            name, _v1b_side = v1b_override(name, r)
            if _v1b_side is not None:
                _m = lesions == r["lesion_mask_idx"]
                _g = gt[_m]; _g = _g[_g > 0]
                _gid = int(np.bincount(_g).argmax()) if _g.size else 0
                print(f"  [v1b] {cid} les{int(r['lesion_mask_idx'])} {_v1b_old} -> {name} (gt id {_gid})", flush=True)
                if V1B_DUMP:
                    with open(V1B_DUMP, "a") as _fh:
                        _fh.write(json.dumps({"case": cid, "les": int(r["lesion_mask_idx"]), "old": _v1b_old,
                                              "new": name, "gtid": _gid, "vox": int(_m.sum())}) + "\\n")
            if DUMP_RANK:''', "v1b hook")

open(f"{S}/c5_v1b.py", "w").write(f"# 자동생성: experiments/D1_newdata/v1b_make.py · 원본 c5_location_v2.py md5 {md5}\n" + s)
print("생성: c5_v1b.py")
