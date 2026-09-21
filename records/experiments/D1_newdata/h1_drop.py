#!/usr/bin/env python3
"""h1_drop.py <logs_dir> — 패치필터가 실제로 몇 개를 지웠는지 (팔 × split, 5시드 평균)."""
import sys, os, json, glob
LG = sys.argv[1]
print("| 구성 | split | blob 수 | 제거 | 제거율 | 케이스당 초 |")
print("|---|---|---|---|---|---|")
for tag in ("e9off", "b1on"):
    for sp in ("test", "val"):
        rs = []
        for p in sorted(glob.glob(f"{LG}/pf_{tag}_{sp}_s*.json")):
            try:
                rs.append(json.load(open(p)))
            except Exception:
                pass
        if not rs:
            continue
        nb = sum(r["n_blob"] for r in rs) / len(rs)
        nd = sum(r["n_drop"] for r in rs) / len(rs)
        sec = sum(r["sec_total"] for r in rs) / len(rs) / max(rs[0]["n_cases"], 1)
        print(f"| {tag} | {sp} | {nb:.1f} | {nd:.1f} | {100*nd/max(nb,1e-9):.1f}% | {sec:.1f}s |")
