"""C41 — 앵커가 왜 없나: 혈관 부재인가 노드형성 실패인가 (2026-08-18).

측정된 사실: 참조 혈관마스크 417케이스에서 분기점 타입별 검출률이
  R-P1P2+R-Pcom 41.7% · L-ICA-C6-C7+L-Pcom 42.7% · R-ICA-C6-C7+R-AChA 43.2% ...
로 7종이 60% 미만이고, 이들이 정확히 무너지는 클래스(3.2 OA · 3.4 Pcom · 3.5 AChA)를 정의한다.
케이스의 절반 이상에서 그 클래스를 정의하는 분기점이 입력에 아예 없으므로
3.4/3.5/3.6 이 동일한 피처를 받고, 어떤 분류기도 구분할 수 없다.

원인이 둘이고 처방이 완전히 다르다:
  (가) 두 혈관 중 하나가 마스크에 아예 없다      -> 분할 문제. 앵커 대체값 추정(C43)으로 우회
  (나) 둘 다 있는데 접촉하지 않아 노드가 안 생긴다 -> 추출 문제. 노드 규칙 완화로 바로 회수

두 비율을 나눠 재서 어느 쪽인지 확정한다.
"""
import collections, glob, json, os
from pathlib import Path

import numpy as np
import nibabel as nib

import d9xx_lib as L
import c5_location_v2 as C5

REF = L.TOPANEU_ROOT / "nnunet" / "nnUNet_raw" / "Dataset800_TopAneuVessel417" / "labelsTr"
BP = L.TOPANEU_ROOT / "experiments" / "_c4_bpgraph" / "all_ref"


def main():
    names = L.vessel_dense_names()                 # id -> name
    name2id = {v: k for k, v in names.items()}
    pairs = C5.JUNCTION_PAIRS
    present = collections.Counter()                # 두 혈관 다 마스크에 있음
    detected = collections.Counter()               # 분기점 노드가 잡힘
    tot = 0
    files = sorted(glob.glob(str(BP / "*.json")))
    for i, f in enumerate(files, 1):
        cid = Path(f).stem
        vp = REF / f"{cid}.nii.gz"
        if not vp.exists():
            continue
        tot += 1
        ids = set(int(x) for x in np.unique(np.asanyarray(nib.load(vp).dataobj)) if x != 0)
        have = {names[i_] for i_ in ids if i_ in names}
        nodes = json.load(open(f)).get("nodes", [])
        seen = set()
        for nd in nodes:
            cs = set(nd["classes"])
            for k, p in enumerate(pairs):
                if set(p) <= cs:
                    seen.add(k)
        for k, (a, b) in enumerate(pairs):
            if a in have and b in have:
                present[k] += 1
            if k in seen:
                detected[k] += 1
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    rows = []
    for k, (a, b) in enumerate(pairs):
        pr, dt = present[k] / tot, detected[k] / tot
        # 두 혈관이 다 있는 케이스 중 노드가 잡힌 비율 = 추출 효율
        eff = detected[k] / present[k] if present[k] else 0.0
        rows.append({"pair": f"{a} + {b}", "present": pr, "detected": dt, "extract_eff": eff,
                     "lost_to_segmentation": 1 - pr, "lost_to_extraction": pr - dt})
    rows.sort(key=lambda r: r["detected"])
    print(f"\n참조 마스크 {tot}케이스\n")
    print(f"{'분기점':<34}{'혈관존재':>9}{'노드검출':>9}{'추출효율':>9}{'분할손실':>9}{'추출손실':>9}")
    for r in rows:
        print(f"{r['pair']:<34}{r['present']*100:>8.1f}%{r['detected']*100:>8.1f}%"
              f"{r['extract_eff']*100:>8.1f}%{r['lost_to_segmentation']*100:>8.1f}%"
              f"{r['lost_to_extraction']*100:>8.1f}%")
    low = [r for r in rows if r["detected"] < 0.60]
    seg = float(np.mean([r["lost_to_segmentation"] for r in low])) if low else 0
    ext = float(np.mean([r["lost_to_extraction"] for r in low])) if low else 0
    print(f"\n[검출률 60% 미만 {len(low)}종 평균] 분할손실 {seg*100:.1f}% · 추출손실 {ext*100:.1f}%")
    print("  분할손실이 크면 -> 앵커 대체값 추정(C43)으로 우회해야 한다")
    print("  추출손실이 크면 -> 노드 형성 규칙 완화로 바로 회수된다")
    out = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis" / "c41_anchor_diag.json"
    json.dump({"n_cases": tot, "rows": rows,
               "low_detect_seg_loss": seg, "low_detect_ext_loss": ext},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
