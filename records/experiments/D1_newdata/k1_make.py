#!/usr/bin/env python3
"""K1 — 두 학습표 모델 확률 평균을 넣은 c5 복사본(c5_k1.py)을 원본에서 기계적으로 만든다.

기준 모델(개정판 하이브리드표) 학습 직후, TOPANEU_K1_FEAT2 가 가리키는 두 번째 표(예측혈관표 · V3-P)로 같은 설정의
모델을 하나 더 학습하고 model["clf"] 를 c5 의 ProbAvg([기준, 두번째]) 로 바꾼다. 두 표는 같은 271병변·같은 라벨이라
classes_ 와 사전확률(pri)이 같다(assert). 이후 β·확신게이트·gC 2등 조각은 **평균 확률** 위에서 그대로 동작한다.
V3-M(행 병합)과 다르다 — 각 모델은 자기 조건만 배우고 결정만 섞는다. TOPANEU_K1_FEAT2 가 비면 원본과 완전히 같다.
"""
import hashlib
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
S = f"{R}/code/sblee/nnunet/scripts"


def sub(src, old, new, what):
    assert src.count(old) == 1, f"{what}: 원문이 {src.count(old)}번"
    return src.replace(old, new)


s = open(f"{S}/c5_location_v2.py").read()
md5 = hashlib.md5(s.encode()).hexdigest()
s = sub(s, '''    model = fit_model(train_rows, ves_axis, kind=args.model, k=args.k,
                      mirror=not args.no_mirror, balance=not args.no_balance,
                      synth_rows=synth, synth_repeat=args.synth_repeat)
    if args.save_model:''', '''    model = fit_model(train_rows, ves_axis, kind=args.model, k=args.k,
                      mirror=not args.no_mirror, balance=not args.no_balance,
                      synth_rows=synth, synth_repeat=args.synth_repeat)
    # ── K1 · 두 학습표 모델 확률 평균 (experiments/D1_newdata/k1_make.py 가 삽입) ──
    _k1_feat2 = os.environ.get("TOPANEU_K1_FEAT2", "")
    if _k1_feat2:
        assert args.model == "rf", "K1 은 rf 전용"
        _rows2 = json.load(open(_k1_feat2))
        _m2 = fit_model(_rows2, ves_axis, kind=args.model, k=args.k,
                        mirror=not args.no_mirror, balance=not args.no_balance,
                        synth_rows=synth, synth_repeat=args.synth_repeat)
        assert list(_m2["clf"].classes_) == list(model["clf"].classes_), "두 표 클래스 불일치"
        assert np.allclose(_m2["pri"], model["pri"]), "두 표 사전확률 불일치"
        model["clf"] = ProbAvg([model["clf"], _m2["clf"]])
        print(f"[c5][K1] 확률 평균: 기준표 {len(train_rows)}행 + {os.path.basename(_k1_feat2)} {len(_rows2)}행", flush=True)
    if args.save_model:''', "k1 block")
open(f"{S}/c5_k1.py", "w").write(f"# 자동생성: experiments/D1_newdata/k1_make.py · 원본 c5_location_v2.py md5 {md5}\n" + s)
print("생성: c5_k1.py")
