#!/usr/bin/env python3
"""g2_seeds.py <scores_dir> <on_tag> <off_tag> [title] — 새 eval 5시드 (on−off) 표 + 고정 규칙 판정.
파일명: <scores_dir>/<tag>_<split>_s<seed>.json (neweval.py 출력)."""
import sys, os, json
M = ["PRECISION","RECALL","F1","MCC","DICE","VOLSIM","HD95"]; HI = {"HD95":-1}
NEED = 5   # 공식 7지표 중 최소 개선 수 (구 규칙 4/6 과 같은 엄격도)
SD, ON, OFF = sys.argv[1:4]; TITLE = sys.argv[4] if len(sys.argv) > 4 else f"{ON} vs {OFF}"
def L(p):
    if not os.path.exists(p): return None
    s = open(p).read(); i = len(s)
    while True:
        i = s.rfind('{', 0, i)
        if i < 0: return None
        try:
            d = json.loads(s[i:])
            if isinstance(d, dict) and "label" in d: return d
        except Exception: pass
def v(d,m):
    x = d["new"].get(m); return float("nan") if x is None else float(x)
def vo(d,m):
    x = (d.get("old") or {}).get(m); return None if x is None else float(x)
print(f"# {TITLE} — 새 공식 eval · {ON} − {OFF} · seed 0~4\n")
verdict = {}
for sp in ("test","val"):
    rows = {m: [] for m in M}; seeds = []; old = []
    for sd in range(5):
        on, off = L(f"{SD}/{ON}_{sp}_s{sd}.json"), L(f"{SD}/{OFF}_{sp}_s{sd}.json")
        if not on or not off: continue
        seeds.append(sd)
        for m in M: rows[m].append(v(on,m)-v(off,m))
        a, b = vo(on,"MCC"), vo(off,"MCC")
        if a is not None and b is not None: old.append(a-b)
    if not seeds: print(f"[{sp}] 결과 없음\n"); continue
    print(f"## {sp} · 시드 {seeds}\n")
    print("| 지표 | " + " | ".join(f"s{s}" for s in seeds) + " | 평균Δ | 개선 |"); print("|---|" + "---|"*(len(seeds)+2))
    n = 0; dm = 0.0
    for m in M:
        mean = sum(rows[m])/len(rows[m]); imp = mean*HI.get(m,1) > 0; n += imp
        if m == "MCC": dm = mean
        print(f"| {m} | " + " | ".join(f"{x:+.4f}" for x in rows[m]) + f" | {mean:+.4f} | {'○' if imp else '×'} |")
    ok = n >= NEED and dm >= 0; verdict[sp] = ok
    print(f"\n→ 개선 {n}/{len(M)} · 평균ΔMCC {dm:+.4f} · **{'충족' if ok else '미충족'}**")
    if old: print(f"(참고: 구 eval ÷52 MCC 평균Δ {sum(old)/len(old):+.4f})")
    print()
if verdict:
    fin = "채택" if all(verdict.values()) else "미채택"
    print(f"**판정 ({'·'.join(k+':'+('충족' if ok else '미충족') for k,ok in verdict.items())}) → {fin}**")
