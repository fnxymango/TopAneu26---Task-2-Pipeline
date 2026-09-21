#!/usr/bin/env python3
"""K7 — 학습표 N개 모델의 확률 평균(c5_k7.py 자동생성). K1(2표)의 일반화.

K1 은 [기준 하이브리드표 + 예측혈관표] 두 모델의 확률을 평균했다(시드 0~9 합 단위 오른 13 · 내린 1 ·
test TP +18 · FP +8 · RECALL 10/10 시드 개선, 다만 MCC·PRECISION 은 하락).
K7 은 여기에 **검출 blob OOF 표**(V4-D 1단계 산출 · 추론 조건과 같은 입력: 검출 blob + 예측 혈관)를 세 번째 모델로 넣는다.
V4-D 는 '표 교체/병합' 으로는 관문 미달이었지만(Δmacro +0.008/+0.016 < +0.02), 같은 표를 **결정 평균**의 한 축으로
쓰는 것은 다른 장치다 — K1 에서 V3-P 표가 단독 채택은 안 됐어도 평균에서는 적중을 늘린 전례가 있다.
TOPANEU_K7_FEATS 에 쉼표로 표 경로들을 준다(비면 원본과 동일). gt_loc 없는 행(환각)은 제외한다.
"""
import hashlib
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
S = f"{R}/code/sblee/nnunet/scripts"
s = open(f"{S}/c5_location_v2.py").read()
md5 = hashlib.md5(s.encode()).hexdigest()
old = '''    model = fit_model(train_rows, ves_axis, kind=args.model, k=args.k,
                      mirror=not args.no_mirror, balance=not args.no_balance,
                      synth_rows=synth, synth_repeat=args.synth_repeat)
    if args.save_model:'''
new = '''    model = fit_model(train_rows, ves_axis, kind=args.model, k=args.k,
                      mirror=not args.no_mirror, balance=not args.no_balance,
                      synth_rows=synth, synth_repeat=args.synth_repeat)
    # ── K7 · 학습표 N개 모델 확률 평균 (experiments/D1_newdata/k7_make.py 가 삽입) ──
    _k7 = [p for p in os.environ.get("TOPANEU_K7_FEATS", "").split(",") if p]
    if _k7:
        assert args.model == "rf", "K7 은 rf 전용"
        _ms = [model["clf"]]
        for _p in _k7:
            _rows2 = [r for r in json.load(open(_p)) if r.get("gt_loc")]
            _m2 = fit_model(_rows2, ves_axis, kind=args.model, k=args.k,
                            mirror=not args.no_mirror, balance=not args.no_balance,
                            synth_rows=synth, synth_repeat=args.synth_repeat)
            assert set(_m2["clf"].classes_) <= set(model["clf"].classes_), "추가 표에 기준표에 없는 클래스"
            _ms.append(_m2["clf"])
            print(f"[c5][K7] + {os.path.basename(_p)} {len(_rows2)}행 · 클래스 {len(_m2['clf'].classes_)}", flush=True)
        model["clf"] = ProbAvg(_ms)
        print(f"[c5][K7] 확률 평균 모델 {len(_ms)}개 (기준표 {len(train_rows)}행 포함)", flush=True)
    if args.save_model:'''
assert s.count(old) == 1
open(f"{S}/c5_k7.py", "w").write(f"# 자동생성: experiments/D1_newdata/k7_make.py · 원본 c5_location_v2.py md5 {md5}\n" + s.replace(old, new))
print("c5_k7.py 생성")
