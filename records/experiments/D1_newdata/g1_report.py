#!/usr/bin/env python3
"""G1 채점 파일(앞에 디버그 잡음이 섞여도) 을 읽어 gC on/off 를 새·구 eval 로 나란히 보여준다."""
import json, os, sys
G = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/G1_gc_neweval"
M = ["PRECISION","RECALL","MCC","DICE","VOLSIM","HD95"]
def L(t):
    p = f"{G}/scores/{t}.json"
    if not os.path.exists(p): p = f"{G}/scores_par/{t}.json"
    if not os.path.exists(p): return None
    s = open(p).read()
    # 앞쪽 디버그 잡음을 건너뛰기: 끝에서부터 '{' 위치마다 파싱 시도, 'label' 키가 있는 첫 성공을 채택
    i = len(s)
    while True:
        i = s.rfind('{', 0, i)
        if i < 0: return None
        try:
            d = json.loads(s[i:])
            if isinstance(d, dict) and "label" in d: return d
        except Exception:
            pass
verdict = {}
for stage, pre, name in (("A","","P3 검출기 위 (예비)"), ("B","b1","번들 검출기 B1 위 (본판정)")):
    for sp in ("test","val"):
        on, off = L(f"{pre}gcON_{sp}"), L(f"{pre}gcOFF_{sp}")
        if not on or not off: continue
        print(f"\n[{name} · {sp} {on['n_cases']}건 · seed3]")
        print(f"{'지표':10s} {'새eval off':>11s} {'새eval on':>10s} {'Δ':>8s} 개선  valid off→on   {'구eval off':>10s} {'구eval on':>9s} {'Δ':>8s}")
        imp = 0; mcc_ok = None
        for m in M:
            a, b = off['new'][m], on['new'][m]; d = b - a
            better = (d < 0) if m == "HD95" else (d > 0); imp += better
            if m == "MCC": mcc_ok = d >= 0
            oa, ob = off.get('old',{}).get(m), on.get('old',{}).get(m)
            o = f"{oa:10.4f} {ob:9.4f} {ob-oa:+8.4f}" if oa is not None else "        (없음)"
            print(f"{m:10s} {a:11.4f} {b:10.4f} {d:+8.4f}  {'○' if better else '×'}   {off['new']['count_valid_'+m]:2d}→{on['new']['count_valid_'+m]:2d}      {o}")
        print(f"→ 개선 {imp}/6 · MCC 비악화 {mcc_ok}   (규칙: ≥4 AND MCC 비악화)")
        verdict[(stage, sp)] = (imp >= 4 and bool(mcc_ok))
if verdict: print("\n판정 요약:", {f"{k[0]}-{k[1]}": ("유지" if v else "불충족") for k, v in verdict.items()})
