#!/usr/bin/env python
"""약한 클래스 전용 채점 (2026-08-25).

전체 cov.MCC 는 시드 산포 ±0.02 라 개입 효과가 묻힌다. E16 이 특정한
**약한 클래스 22개(test GT 병변 40/86)** 에서 TP 가 늘었는지를 따로 본다.
집계 MCC 와 함께 보되, "어디가 고쳐졌나" 는 이쪽이 답한다.
"""
import json, glob, os, re, sys, collections
import numpy as np

A = os.path.join(os.environ["TOPANEU_ROOT"], "code/sblee/nnunet/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d9xx_lib as L


def weak_sets():
    d = json.load(open(os.path.join(A, "weak_classes.json")))
    return set(d["weak"]), set(d["strong"])


def per_class(tag, split="test"):
    """태그의 시드별 per-case 파일에서 클래스별 TP/FN/FP 를 모은다."""
    id2name, _ = L.official_location_names()
    out = {}
    for f in sorted(glob.glob(os.path.join(A, f"c5_percase_{split}_{tag}_s?.json"))):
        sd = int(re.search(r"_s(\d)\.json$", f).group(1))
        d = json.load(open(f))
        pres = d["present"]
        tp = collections.Counter(); fn = collections.Counter(); fp = collections.Counter()
        for pc in d["per_case"]:
            for c in pres:
                tp[c] += pc.get(f"TP_{c}", 0)
                fn[c] += pc.get(f"FN_{c}", 0)
                fp[c] += pc.get(f"FP_{c}", 0)
        out[sd] = {id2name.get(c, f"cls{c}"): (tp[c], fn[c], fp[c]) for c in pres}
    return out


def summarize(tag, split="test"):
    weak, strong = weak_sets()
    pc = per_class(tag, split)
    if not pc:
        return None
    rows = []
    for sd, m in sorted(pc.items()):
        wt = sum(v[0] for k, v in m.items() if k in weak)
        wg = sum(v[0] + v[1] for k, v in m.items() if k in weak)
        st = sum(v[0] for k, v in m.items() if k in strong)
        sg = sum(v[0] + v[1] for k, v in m.items() if k in strong)
        fp = sum(v[2] for v in m.values())
        rows.append((sd, wt, wg, st, sg, fp))
    return rows


def main():
    split = os.environ.get("WEAK_SPLIT", "test")
    tags = sys.argv[1:]
    print(f"[약한클래스 채점 · {split}]  약한 22개 / 강한 14개 (E16)")
    print(f"  {'설정':<26}{'약한TP':>9}{'약한GT':>8}{'강한TP':>9}{'강한GT':>8}{'FP':>7}")
    base = None
    for t in tags:
        rows = summarize(t, split)
        if not rows:
            print(f"  {t:<26} 데이터 없음"); continue
        a = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
        m = a.mean(axis=0)
        line = f"  {t:<26}{m[0]:>9.1f}{m[1]:>8.0f}{m[2]:>9.1f}{m[3]:>8.0f}{m[4]:>7.1f}"
        if base is not None:
            line += f"   약한Δ {m[0]-base[0]:+.1f}  강한Δ {m[2]-base[2]:+.1f}"
        else:
            base = m
            line += "   ← 기준"
        print(line)


if __name__ == "__main__":
    main()
