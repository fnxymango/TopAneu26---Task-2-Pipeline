#!/usr/bin/env python3
"""T3 — 비낭형(방추·박리) 학습 행에 표본 가중을 준 c5 복사본(c5_t3.py)을 원본에서 기계적으로 만든다.

왜: train 방추형 34 · 박리형 7 이 낭형 230 에 묻힌다. 방추형 OOF 정확도 53%(낭형 73%). 방추형은 여러 분절에 걸쳐도
GT 가 몸통으로 라벨하는데, 표본이 적어 RF 가 그 규약을 못 배우고 '닿은 곳=접합' 으로 답한다.
가중: 비낭형 행 w = N낭형/N비낭형 (학습표에서 계산 · 유형 균형, 튜닝값 아님). 미러 행도 같은 가중. RF 의 class_weight='balanced'
위에 sample_weight 로 곱해진다. 행 복제는 쓰지 않는다 — 사전확률(pri) 이 바뀌어 β 보정까지 흔들리기 때문.
유형은 **학습 행에만** 필요하다(GT type_masks → k6_type_feat.json). 추론은 원본과 같다 → 배포는 모델 파일 교체만.
TOPANEU_T3=1 일 때만 동작. 끄면 원본과 완전히 같다.
"""
import hashlib
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
S = f"{R}/code/sblee/nnunet/scripts"; D = f"{R}/experiments/D1_newdata"


def sub(src, old, new, what):
    assert src.count(old) == 1, f"{what}: 원문이 {src.count(old)}번"
    return src.replace(old, new)


s = open(f"{S}/c5_location_v2.py").read()
md5 = hashlib.md5(s.encode()).hexdigest()
s = sub(s, "def fit_model(rows, ves_axis,", f'''# ── T3 · 비낭형 표본 가중 (experiments/D1_newdata/t3_make.py 가 삽입) ─────────────────────
T3_ON = os.environ.get("TOPANEU_T3", "0") == "1"
_T3_TYPES = None


def _t3_weights(rows):
    """학습 행별 가중. 유형표에 없는 행(합성 등)은 1."""
    global _T3_TYPES
    if _T3_TYPES is None:
        _T3_TYPES = {{k: int(v["type"] in (2, 3)) for k, v in json.load(open("{D}/k6_type_feat.json")).items()}}
    flags = [_T3_TYPES.get(f"{{r.get('case')}}|{{r.get('lesion_mask_idx')}}") for r in rows]
    n_ns = sum(1 for f in flags if f == 1); n_s = sum(1 for f in flags if f == 0)
    w_ns = (n_s / n_ns) if n_ns else 1.0
    return [w_ns if f == 1 else 1.0 for f in flags], w_ns, n_ns


def fit_model(rows, ves_axis,''', "t3 globals")
s = sub(s, '''    X, y = [], []
    for r in rows:
        X.append(row_to_vec(r, ves_axis, mirror=False)); y.append(r["gt_loc"])
        if mirror:
            X.append(row_to_vec(r, ves_axis, mirror=True)); y.append(mirror_name(r["gt_loc"]))''', '''    X, y = [], []
    _sw = []
    _rw, _w_ns, _n_ns = _t3_weights(rows) if T3_ON else ([1.0] * len(rows), 1.0, 0)
    for r, _w in zip(rows, _rw):
        X.append(row_to_vec(r, ves_axis, mirror=False)); y.append(r["gt_loc"]); _sw.append(_w)
        if mirror:
            X.append(row_to_vec(r, ves_axis, mirror=True)); y.append(mirror_name(r["gt_loc"])); _sw.append(_w)''', "t3 weights loop")
s = sub(s, '''                X.append(row_to_vec(s, ves_axis, mirror=False)); y.append(s["gt_loc"])''',
        '''                X.append(row_to_vec(s, ves_axis, mirror=False)); y.append(s["gt_loc"]); _sw.append(1.0)''', "t3 synth")
s = sub(s, '''    if kind != "knn":
        clf = build_clf(kind, X, y)''', '''    if kind != "knn":
        assert not (T3_ON and SMOTE_N > 0), "T3 와 SMOTE 동시 사용 불가"
        clf = build_clf(kind, X, y, sw=np.array(_sw) if T3_ON else None)
        if T3_ON:
            print(f"  [T3] 비낭형 {_n_ns}행 가중 {_w_ns:.2f}", flush=True)''', "t3 fit")
s = sub(s, "def build_clf(kind, X, y):", "def build_clf(kind, X, y, sw=None):", "t3 sig")
s = sub(s, '''                                      class_weight="balanced", random_state=_seed(),
                                      n_jobs=-1).fit(X, y)''', '''                                      class_weight="balanced", random_state=_seed(),
                                      n_jobs=-1).fit(X, y, sample_weight=sw)''', "t3 rf fit")
open(f"{S}/c5_t3.py", "w").write(f"# 자동생성: experiments/D1_newdata/t3_make.py · 원본 c5_location_v2.py md5 {md5}\n" + s)
print("생성: c5_t3.py")
