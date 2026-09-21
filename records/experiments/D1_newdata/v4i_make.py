#!/usr/bin/env python3
"""V4-I — ICA 원위 기하 룰(예측 혈관 조건 재보정)을 넣은 c5 복사본(c5_v4i.py)을 원본에서 기계적으로 만든다.

설계·τ 선택은 v4i_screen.py(train OOF · 예측 혈관 vespp_train)에서 결과 보기 전 고정 규칙으로 했다 → v4i_rule.json.
  RF 1등이 ICA 원위(3.2~3.7)이면 blob 이 가장 가까운(≤5mm) ICA-C6-C7 축에서 t(측지 위치)를 재고
    t < 0.493 → C6: OA 접촉(dist_mm ≤ τ) → 3.2 / 아니면 3.3
    t < 0.814 → C7: Pcom 접촉 → 3.4 / AChA 접촉 → 3.5 / 아니면 3.6
    그 외     → 3.7
  쪽은 가까운 축. 축이 없거나 5mm 안에 없으면 RF 그대로. 절단 0.493/0.814 는 V2-2(fine 라벨 train) 추정.
  TOPANEU_V4I=0 이면 원본과 완전히 같다.
"""
import hashlib, json
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
S = f"{R}/code/sblee/nnunet/scripts"
D = f"{R}/experiments/D1_newdata"
rule = json.load(open(f"{D}/v4i_rule.json"))
assert rule["gate"] and rule["mode"] == "M1", rule


def sub(src, old, new, what):
    assert src.count(old) == 1, f"{what}: 원문이 {src.count(old)}번"
    return src.replace(old, new)


s = open(f"{S}/c5_location_v2.py").read()
md5 = hashlib.md5(s.encode()).hexdigest()
s = sub(s, "def cmd_eval(args):", f'''# ── V4-I · ICA 원위 기하 룰 (experiments/D1_newdata/v4i_make.py 가 삽입) ─────────────────
V4I_ON = os.environ.get("TOPANEU_V4I", "0") == "1"
V4I_TAU = float(os.environ.get("TOPANEU_V4I_TAU", "{rule['tau']}"))
V4I_C1, V4I_C2 = 0.493, 0.814
V4I_DUMP = os.environ.get("TOPANEU_V4I_DUMP", "")
_V4I_AX = {{}}
_V4I_CODE = re.compile(r"^(?:[RL]-)?(3\\.[2-7])\\s")


def _v4i_axes(cid, ves, spacing):
    if cid not in _V4I_AX:
        import sys as _s
        _s.path.insert(0, "{D}")
        import ica_split_lib as _IL
        _V4I_AX.clear()
        _V4I_AX[cid] = {{sd: _IL.side_axis(ves, spacing, sd) for sd in ("R", "L")}}
    return _V4I_AX[cid]


def v4i_override(name, r, cid, ves, spacing, lesions):
    """(새 이름, 정보 또는 None)"""
    if not V4I_ON or not name or not _V4I_CODE.match(name):
        return name, None
    les = lesions == r["lesion_mask_idx"]
    per = {{}}
    for sd, ax in _v4i_axes(cid, ves, spacing).items():
        if ax is None:
            continue
        lc = les[ax["sl"]]
        if not lc.any():
            continue
        d = ndimage.distance_transform_edt(~lc, sampling=spacing)
        dmin = float(np.nanmin(np.where(ax["body"], d, np.inf)))
        if dmin > 5.0:
            continue
        near = ax["body"] & (d <= dmin + 1.0)
        per[sd] = (dmin, float(np.nanmedian(ax["t"][near])))
    if not per:
        return name, None
    sd = min(per, key=lambda z: per[z][0]); t = per[sd][1]
    dm = r.get("dist_mm") or {{}}
    touch = lambda v: dm.get(f"{{sd}}-{{v}}") is not None and dm.get(f"{{sd}}-{{v}}") <= V4I_TAU
    if t < V4I_C1:
        k = "3.2" if touch("OA") else "3.3"
    elif t < V4I_C2:
        k = "3.4" if touch("Pcom") else ("3.5" if touch("AChA") else "3.6")
    else:
        k = "3.7"
    new = _V4I_NAMES_TBL[f"{{sd}}-{{k}}"]
    return new, dict(side=sd, t=round(t, 3), dmin=round(per[sd][0], 2))


def cmd_eval(args):''', "v4i block")

s = sub(s, '''            name = predict_one_blend(mdl, r, args.beta, cp, args.crop_w)
            if DUMP_RANK:''', '''            name = predict_one_blend(mdl, r, args.beta, cp, args.crop_w)
            _v4i_old = name
            name, _v4i_info = v4i_override(name, r, cid, ves, spacing, lesions)
            if _v4i_info is not None:
                _m = lesions == r["lesion_mask_idx"]
                _g = gt[_m]; _g = _g[_g > 0]
                _gid = int(np.bincount(_g).argmax()) if _g.size else 0
                if V4I_DUMP:
                    with open(V4I_DUMP, "a") as _fh:
                        _fh.write(json.dumps({"case": cid, "les": int(r["lesion_mask_idx"]), "old": _v4i_old,
                                              "new": name, "gtid": _gid, "vox": int(_m.sum()), **_v4i_info}) + "\\n")
            if DUMP_RANK:''', "v4i hook")

names = {v.split()[0]: v for v in json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["location_classes"].values()}
s = sub(s, "V4I_DUMP = os.environ", f"_V4I_NAMES_TBL = {json.dumps({k: v for k, v in names.items() if '3.' in k}, ensure_ascii=False)}\nV4I_DUMP = os.environ", "names")
open(f"{S}/c5_v4i.py", "w").write(f"# 자동생성: experiments/D1_newdata/v4i_make.py · 원본 c5_location_v2.py md5 {md5}\n" + s)
print("생성: c5_v4i.py")
