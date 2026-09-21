"""C7 — 동맥류 검출 후처리 스윕 (재학습 없이 동작점 조정).

근거(oracle_location_report_417.md §4): GT 병변에서 최근접 혈관까지 거리는
중앙값 0.30mm / 95퍼센타일 0.55mm. 즉 진짜 동맥류는 사실상 혈관에 붙어 있다.
따라서 혈관에서 멀리 떨어진 예측 성분은 거의 확실히 위양성이고, 재학습 없이 걸러낼 수 있다.
(리포트 자체가 이 항목을 "FP 필터용"으로 달아뒀다.)

또 하나: 아주 작은 성분도 FP 비중이 높다. 다만 동맥류는 원래 수~십 voxel 초소형이라
크기 임계를 세게 걸면 진짜 병변을 잃는다 — 그래서 스윕으로 교환비를 직접 본다.

지표는 voxel Dice가 아니라 **병변 단위**로 본다(PROJECT_RULES.md §6):
  - lesion sensitivity: GT 병변 중 pred와 1voxel이라도 겹친 비율
  - FP/case: 어떤 GT와도 안 겹친 pred 성분 수

사용:
  python c7_detect_postproc.py --aneu-dir <..> --vessel-dir <..> --split val \
      [--min-vox 0,5,10,20,40] [--max-dist 999,5,3,2,1] [--save-best <dir>]
"""
import argparse, json, os, itertools
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

ST = np.ones((3, 3, 3), dtype=bool)
ROOT = Path(os.environ.get("TOPANEU_ROOT", os.path.expanduser("~/topaneu_sblee")))
LAB = ROOT / "nnunet" / "nnUNet_raw" / "Dataset720_TopAneuBinary417" / "labelsTr"
SPLIT = ROOT / "dataset" / "TopAneu" / "dataset_split.json"
OUT = ROOT / "code" / "sblee" / "nnunet" / "analysis"


def case_ids(split):
    d = json.load(open(SPLIT))["splits"]
    return d[split]


def component_stats(aneu, ves, spacing):
    """예측 성분별 (라벨, voxel수, 혈관까지 최소거리mm)."""
    lab, n = ndi.label(aneu > 0, structure=ST)
    if n == 0:
        return lab, []
    # 혈관까지의 거리장 — 혈관이 아예 없으면 전부 inf 취급
    if (ves > 0).any():
        dist = ndi.distance_transform_edt(~(ves > 0), sampling=spacing)
    else:
        dist = np.full(aneu.shape, np.inf)
    out = []
    for l in range(1, n + 1):
        m = lab == l
        out.append((l, int(m.sum()), float(dist[m].min())))
    return lab, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aneu-dir", required=True)
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--min-vox", default="0,5,10,20,40")
    ap.add_argument("--max-dist", default="999,5,3,2,1")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-best", default=None, help="최고 설정으로 필터된 마스크를 이 폴더에 저장")
    ap.add_argument("--force-cfg", default=None, help="'minvox,maxdist' 형태로 설정 고정(저장용)")
    args = ap.parse_args()

    aneu_dir = Path(args.aneu_dir); ves_dir = Path(args.vessel_dir)
    tag = args.tag or f"{aneu_dir.name}_{args.split}"
    ids = case_ids(args.split)
    MINV = [int(x) for x in args.min_vox.split(",")]
    MAXD = [float(x) for x in args.max_dist.split(",")]

    # 케이스별로 한 번만 성분 통계를 뽑고, 그 위에서 모든 설정을 평가한다(재계산 방지)
    cache = []
    for i, cid in enumerate(ids, 1):
        ap_ = aneu_dir / f"{cid}.nii.gz"; vp = ves_dir / f"{cid}.nii.gz"; gp = LAB / f"{cid}.nii.gz"
        if not (ap_.exists() and vp.exists() and gp.exists()):
            continue
        ai = nib.load(ap_)
        aneu = np.asanyarray(ai.dataobj)
        ves = np.asanyarray(nib.load(vp).dataobj)
        gt = np.asanyarray(nib.load(gp).dataobj) > 0
        spacing = np.array(ai.header.get_zooms()[:3], dtype=float)
        lab, comps = component_stats(aneu, ves, spacing)
        glab, gn = ndi.label(gt, structure=ST)
        # 성분별로 어떤 GT 병변과 겹치는지 미리 계산
        hits = []
        for (l, nv, d) in comps:
            m = lab == l
            g_ids = set(int(x) for x in np.unique(glab[m]) if x > 0)
            hits.append((l, nv, d, g_ids))
        cache.append({"case": cid, "gn": int(gn), "comps": hits,
                      "path": str(ap_), "affine": ai.affine, "header": ai.header,
                      "lab": lab if args.save_best else None})
        if i % 20 == 0 or i == len(ids):
            print(f"  통계 {i}/{len(ids)}", flush=True)

    def evaluate(minv, maxd):
        det, tot, fps = 0, 0, []
        for c in cache:
            tot += c["gn"]
            kept = [h for h in c["comps"] if h[1] >= minv and h[2] <= maxd]
            found = set()
            fp = 0
            for (l, nv, d, g_ids) in kept:
                if g_ids:
                    found |= g_ids
                else:
                    fp += 1
            det += len(found); fps.append(fp)
        return {"min_vox": minv, "max_dist_mm": maxd,
                "lesion_sensitivity": det / tot if tot else None,
                "lesions_detected": det, "lesions_total": tot,
                "fp_per_case": float(np.mean(fps)), "fp_total": int(np.sum(fps))}

    rows = [evaluate(a, b) for a, b in itertools.product(MINV, MAXD)]
    rows.sort(key=lambda r: (-r["lesion_sensitivity"], r["fp_per_case"]))

    print(f"\n=== {tag} — 검출 후처리 스윕 ({len(cache)}케이스) ===")
    print(f"{'min_vox':>8} {'max_dist':>9} {'민감도':>16} {'FP/case':>8} {'FP총계':>7}")
    for r in sorted(rows, key=lambda r: (r["min_vox"], r["max_dist_mm"])):
        print(f"{r['min_vox']:>8} {r['max_dist_mm']:>9.1f} "
              f"{r['lesion_sensitivity']:.3f} ({r['lesions_detected']:>2}/{r['lesions_total']}) "
              f"{r['fp_per_case']:>8.2f} {r['fp_total']:>7}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": tag, "aneu_dir": str(aneu_dir), "vessel_dir": str(ves_dir),
               "split": args.split, "n_cases": len(cache), "grid": rows},
              open(OUT / f"c7_detect_sweep_{tag}.json", "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {OUT / f'c7_detect_sweep_{tag}.json'}")

    if args.save_best:
        minv, maxd = (int(args.force_cfg.split(",")[0]), float(args.force_cfg.split(",")[1])) \
            if args.force_cfg else (rows[0]["min_vox"], rows[0]["max_dist_mm"])
        sd = Path(args.save_best); sd.mkdir(parents=True, exist_ok=True)
        print(f"[저장] 필터 적용 min_vox={minv} max_dist={maxd} -> {sd}")
        for c in cache:
            keep = [h[0] for h in c["comps"] if h[1] >= minv and h[2] <= maxd]
            m = np.isin(c["lab"], keep).astype(np.int16)
            nib.save(nib.Nifti1Image(m, c["affine"], c["header"]), sd / f"{c['case']}.nii.gz")


if __name__ == "__main__":
    main()
