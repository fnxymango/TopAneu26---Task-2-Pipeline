"""동맥류 위치라벨 규칙기반 할당 — 오라클 상한 측정 (417케이스 버전).

oracle_location_assign.py(98케이스, Dataset510/600)를 417케이스(Dataset900/800)로 이식.
로직은 원본과 동일 — 차이는 경로와 라벨 클래스 수(43 vs 36)뿐.

질문: "stage1 동맥류 seg → 가장 가까운 vessel skeleton의 클래스 = 그 동맥류의 클래스"
이 규칙이 GT 입력(완벽한 동맥류 마스크 + 완벽한 vessel 마스크)에서 몇 %를 맞추나?
D9xx의 규칙기반 baseline을 확정하기 위한 사전 검증.

입력: Dataset900(동맥류, 43 위치클래스) GT + Dataset800(vessel, 36클래스) GT
출력:
  analysis/lesion_vessel_features_417.json
  analysis/oracle_location_report_417.md
"""
import json, os, sys, time, collections
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

BASE = Path(os.environ.get("TOPANEU_ROOT", os.path.expanduser("~/topaneu_sblee")))
RAW = BASE / "nnunet" / "nnUNet_raw"
LOC_DIR = RAW / "Dataset900_TopAneuLoc417"
VES_DIR = RAW / "Dataset800_TopAneuVessel417"
OUT = BASE / "code" / "sblee" / "nnunet" / "analysis"
FEAT_JSON = OUT / "lesion_vessel_features_417.json"
REPORT = OUT / "oracle_location_report_417.md"

MAX_R = 10.0
RADII = [1.0, 2.0, 3.0, 5.0]


def load_labels(d):
    lab = json.load(open(d / "dataset.json"))["labels"]
    return {int(v): k for k, v in lab.items()}


def extract():
    loc_names, ves_names = load_labels(LOC_DIR), load_labels(VES_DIR)
    cases = sorted(p.name[:-7] for p in (LOC_DIR / "labelsTr").glob("*.nii.gz"))
    print(f"[extract] {len(cases)} cases", flush=True)

    rows, skipped, t0 = [], [], time.time()
    for i, case in enumerate(cases, 1):
        li = nib.load(LOC_DIR / "labelsTr" / f"{case}.nii.gz")
        vi = nib.load(VES_DIR / "labelsTr" / f"{case}.nii.gz")
        if li.shape != vi.shape:
            skipped.append((case, f"shape {li.shape} vs {vi.shape}"))
            continue
        if not np.allclose(li.affine, vi.affine, atol=1e-3):
            skipped.append((case, "affine 불일치"))
            continue

        loc = np.asanyarray(li.dataobj)
        ves = np.asanyarray(vi.dataobj)
        spacing = np.array(li.header.get_zooms()[:3], dtype=float)
        vox_mm3 = float(np.prod(spacing))

        lesions, n = ndimage.label(loc > 0, structure=np.ones((3, 3, 3)))
        for lid in range(1, n + 1):
            m = lesions == lid
            nvox = int(m.sum())
            vals, cnts = np.unique(loc[m], return_counts=True)
            gt_id = int(vals[np.argmax(cnts)])

            idx = np.argwhere(m)
            lo = idx.min(0); hi = idx.max(0) + 1
            marg = np.ceil(MAX_R / spacing).astype(int) + 1
            lo2 = np.maximum(lo - marg, 0); hi2 = np.minimum(hi + marg, np.array(loc.shape))
            sl = tuple(slice(a, b) for a, b in zip(lo2, hi2))
            mc, vc = m[sl], ves[sl]

            inside = vc[mc]
            ov = collections.Counter(int(x) for x in inside if x > 0)
            overlap = {ves_names[k]: int(v) for k, v in ov.items()}
            inside_frac = float((inside > 0).sum()) / max(nvox, 1)

            dist = ndimage.distance_transform_edt(~mc, sampling=spacing)
            dmap = {}
            for c in np.unique(vc):
                if c == 0:
                    continue
                sel = (vc == c) & (~mc)
                if not sel.any():
                    dmap[ves_names[int(c)]] = 0.0
                    continue
                d = float(dist[sel].min())
                if d <= MAX_R:
                    dmap[ves_names[int(c)]] = round(d, 3)

            cen = idx.mean(0)
            cen_best, cen_bd = None, None
            for c in np.unique(vc):
                if c == 0:
                    continue
                pts = np.argwhere(vc == c) + lo2
                dd = float(np.sqrt((((pts - cen) * spacing) ** 2).sum(1)).min())
                if cen_bd is None or dd < cen_bd:
                    cen_bd, cen_best = dd, ves_names[int(c)]

            rows.append(dict(
                case=case, lesion=lid, n_vox=nvox, vol_mm3=round(nvox * vox_mm3, 2),
                gt_loc=loc_names[gt_id], gt_loc_id=gt_id,
                inside_vessel_frac=round(inside_frac, 4), overlap=overlap,
                dist_mm=dmap,
                centroid_nearest=cen_best,
                centroid_nearest_mm=round(cen_bd, 3) if cen_bd is not None else None,
            ))
        if i % 20 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}  lesions={len(rows)}  {time.time()-t0:.0f}s", flush=True)

    OUT.mkdir(exist_ok=True, parents=True)
    json.dump({"lesions": rows, "skipped": skipped}, open(FEAT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"[extract] 저장 {FEAT_JSON}  병변 {len(rows)}개, skip {len(skipped)}건", flush=True)
    return rows, skipped


def nearest_class(r):
    if not r["dist_mm"]:
        return None
    md = min(r["dist_mm"].values())
    tied = [c for c, d in r["dist_mm"].items() if d <= md + 1e-6]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda c: r["overlap"].get(c, 0))


def inside_major(r):
    if r["overlap"]:
        return max(r["overlap"], key=r["overlap"].get)
    return nearest_class(r)


def partners(r, R, exclude):
    return frozenset(c for c, d in r["dist_mm"].items() if d <= R and c != exclude)


def loo_accuracy(rows, sig_fn):
    sigs = [sig_fn(r) for r in rows]
    backs = [nearest_class(r) for r in rows]
    ys = [r["gt_loc"] for r in rows]
    glob_major = collections.Counter(ys).most_common(1)[0][0]

    ok = 0
    for i in range(len(rows)):
        tab, btab = collections.Counter(), collections.Counter()
        for j in range(len(rows)):
            if j == i:
                continue
            tab[(sigs[j], ys[j])] += 1
            btab[(backs[j], ys[j])] += 1
        cand = {y: n for (s, y), n in tab.items() if s == sigs[i]}
        if not cand:
            cand = {y: n for (s, y), n in btab.items() if s == backs[i]}
        pred = max(cand, key=cand.get) if cand else glob_major
        ok += (pred == ys[i])
    return ok / len(rows)


def report(rows, skipped):
    L = []
    A = L.append
    n = len(rows)
    A("# 동맥류 위치할당 규칙 — 오라클 상한 (417케이스, GT 동맥류 + GT vessel)\n")
    A(f"> 병변 {n}개 / {len(set(r['case'] for r in rows))}케이스. "
      f"생성: `scripts/oracle_location_assign_417.py`. 캐시: `analysis/lesion_vessel_features_417.json`\n")
    if skipped:
        A(f"> 지오메트리 불일치로 제외: {len(skipped)}건 — {skipped[:5]}\n")

    fr = np.array([r["inside_vessel_frac"] for r in rows])
    A("\n## 1. vessel GT가 동맥류 sac을 포함하는가\n")
    A(f"- 병변 voxel 중 vessel 라벨이 칠해진 비율: 중앙값 **{np.median(fr):.3f}**, "
      f"평균 {fr.mean():.3f} (min {fr.min():.3f} / max {fr.max():.3f})")
    A(f"- 거의 전부 포함(>=0.9)인 병변: **{int((fr>=.9).sum())}/{n}** "
      f"({100*(fr>=.9).mean():.0f}%)")
    A(f"- 거의 미포함(<=0.1): **{int((fr<=.1).sum())}/{n}** ({100*(fr<=.1).mean():.0f}%)")

    A("\n## 2. 위치클래스별 최근접 vessel class 분포\n")
    A("| 위치클래스 (n) | 최근접 vessel class 분포 | 최빈 비율 |")
    A("|---|---|---|")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["gt_loc"]].append(r)
    purity = []
    for loc, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        c = collections.Counter(nearest_class(r) for r in rs)
        top = c.most_common(3)
        share = top[0][1] / len(rs)
        purity.append((loc, len(rs), share))
        s = ", ".join(f"{k}x{v}" for k, v in top)
        A(f"| {loc} ({len(rs)}) | {s} | {share:.0%} |")
    w = sum(p * k for _, k, p in purity) / n
    A(f"\n- **가중 평균 최빈비율(=최근접 클래스만으로 낼 수 있는 상한 근사): {w:.1%}**")

    A("\n### 2b. sac 내부 최빈 vessel class 기준 (어노테이터가 칠한 소속)\n")
    A("| 위치클래스 (n) | sac 내부 최빈 vessel class 분포 | 최빈 비율 |")
    A("|---|---|---|")
    purity2 = []
    for loc, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        c = collections.Counter(inside_major(r) for r in rs)
        top = c.most_common(3)
        share = top[0][1] / len(rs)
        purity2.append((loc, len(rs), share))
        A(f"| {loc} ({len(rs)}) | {', '.join(f'{k}x{v}' for k, v in top)} | {share:.0%} |")
    w2 = sum(p * k for _, k, p in purity2) / n
    A(f"\n- **가중 평균 최빈비율: {w2:.1%}**  (최근접 기준 {w:.1%} 대비 {w2-w:+.1%}p)")

    A("\n## 3. 규칙 변형별 위치클래스 정확도 (leave-one-out)\n")
    A("| 규칙 | 정확도 |")
    A("|---|---|")
    variants = [
        ("V0  centroid 최근접 클래스 (원안)", lambda r: (r["centroid_nearest"],)),
        ("V1  병변표면 최근접 클래스", lambda r: (nearest_class(r),)),
        ("V3  **sac 내부 최빈 vessel class**", lambda r: (inside_major(r),)),
    ]
    for R in RADII:
        variants.append((
            f"V2  최근접 + 반경 {R:g}mm 이웃집합",
            (lambda R: lambda r: (nearest_class(r), partners(r, R, nearest_class(r))))(R),
        ))
    for R in RADII:
        variants.append((
            f"V4  **sac 내부 최빈 + 반경 {R:g}mm 이웃집합**",
            (lambda R: lambda r: (inside_major(r), partners(r, R, inside_major(r))))(R),
        ))
    best = None
    for name, fn in variants:
        acc = loo_accuracy(rows, fn)
        A(f"| {name} | **{acc:.1%}** |")
        if best is None or acc > best[1]:
            best = (name, acc)
    A(f"\n- 최고: **{best[0]} -> {best[1]:.1%}**")
    A(f"- 참고: 최빈 클래스만 찍는 baseline = "
      f"{max(collections.Counter(r['gt_loc'] for r in rows).values())/n:.1%}")

    d0 = np.array([min(r["dist_mm"].values()) if r["dist_mm"] else 99 for r in rows])
    A("\n## 4. 병변-혈관 거리 (FP 필터용)\n")
    A(f"- 최근접 혈관까지 거리: 중앙값 {np.median(d0):.2f}mm, "
      f"95퍼센타일 {np.percentile(d0,95):.2f}mm, 최대 {d0.max():.2f}mm")

    REPORT.write_text("\n".join(L) + "\n")
    print(f"[report] 저장 {REPORT}", flush=True)
    print("\n".join(L[-16:]), flush=True)


if __name__ == "__main__":
    if "--reuse" in sys.argv and FEAT_JSON.exists():
        d = json.load(open(FEAT_JSON))
        rows, skipped = d["lesions"], d["skipped"]
        print(f"[cache] {FEAT_JSON} 재사용, 병변 {len(rows)}개", flush=True)
    else:
        rows, skipped = extract()
    report(rows, skipped)
