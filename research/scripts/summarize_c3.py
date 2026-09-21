#!/usr/bin/env python3
"""C3 — 16칸 비교표 생성 (검출기 x 혈관후처리 x C방법 x split).

읽는 파일 (analysis/):
    d9xx_eval_{split}_realpred_{pp|raw}.json          C1 x A5-2
    d9xx_eval_{split}_realpred_a62_{pp|raw}.json      C1 x A6-2
    d910_eval_{split}_realpred_{pp|raw}.json          C2 x A5-2
    d910_eval_{split}_realpred_a62_{pp|raw}.json      C2 x A6-2

지표 두 종류의 뜻:
    official_div52       조직위 공식 — 52로 나눔. split에 없는 클래스도 분모에 포함되므로
                         천장이 존재한다 (val 33/52=0.635, test 36/52=0.692).
    adjusted_div_present split에 실제로 존재하는 클래스 수로 나눔 — 방법 간 비교용.
"""
import json
import os
from pathlib import Path

A = Path(os.environ["TOPANEU_ROOT"]) / "code" / "sblee" / "nnunet" / "analysis"

DETS = [("A5-2 plain z", ""), ("A6-2 adaptive", "a62_")]
METHODS = [("C1 lookup", "d9xx_eval"), ("C2 kNN", "d910_eval")]
VESSELS = [("pp", "pp"), ("raw", "raw")]


def load(prefix, split, det, ves):
    p = A / f"{prefix}_{split}_realpred_{det}{ves}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    out = ["# C3 — 검출기 교체 대조표 (2x2x2x2 = 16칸)", ""]
    out.append("검출기 동작점 (val 42, 병변 단위, checkpoint_best + TTA off):")
    out.append("")
    out.append("| 검출기 | 병변 민감도 | FP 성분/case | FP 총계 |")
    out.append("|---|---|---|---|")
    out.append("| A5-2 plain z | 0.721 (31/43) | 1.31 | 55 |")
    out.append("| A6-2 adaptive | 0.814 (35/43) | 3.62 | 152 |")
    out.append("")

    for split in ("test", "val"):
        ceil = None
        out.append(f"## {split}")
        out.append("")
        rows = []
        for mname, prefix in METHODS:
            for vlabel, ves in VESSELS:
                for dname, det in DETS:
                    d = load(prefix, split, det, ves)
                    if d is None:
                        rows.append((mname, vlabel, dname, None))
                        continue
                    ceil = d.get("n_present_classes_in_split")
                    rows.append((mname, vlabel, dname, d))
        out.append(f"> 존재 클래스 {ceil}/52 → official_div52 천장 ≈ "
                   f"{(ceil / 52):.3f}" if ceil else "> (결과 없음)")
        out.append("")
        out.append("| C방법 | 혈관 | 검출기 | 예측병변 | MCC(공식/52) | MCC(adjusted) "
                   "| PRECISION(adj) | RECALL(adj) | DICE(adj) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for mname, vlabel, dname, d in rows:
            if d is None:
                out.append(f"| {mname} | {vlabel} | {dname} | — | — | — | — | — | — |")
                continue
            o, a = d["official_div52"], d["adjusted_div_present"]
            out.append(
                f"| {mname} | {vlabel} | {dname} | {d['n_lesions_predicted']} "
                f"| {o['MCC']:.4f} | **{a['MCC']:.4f}** "
                f"| {a['PRECISION']:.4f} | {a['RECALL']:.4f} | {a['DICE']:.4f} |")
        out.append("")

        best = [r for r in rows if r[3]]
        if best:
            b = max(best, key=lambda r: r[3]["adjusted_div_present"]["MCC"])
            out.append(f"**{split} 최고 (adjusted MCC)**: {b[0]} / vessel={b[1]} / {b[2]} "
                       f"→ {b[3]['adjusted_div_present']['MCC']:.4f} "
                       f"(공식 {b[3]['official_div52']['MCC']:.4f})")
            out.append("")

    txt = "\n".join(out)
    print(txt)
    p = A / "c3_comparison.md"
    p.write_text(txt, encoding="utf-8")
    print(f"\n[저장] {p}")


if __name__ == "__main__":
    main()
