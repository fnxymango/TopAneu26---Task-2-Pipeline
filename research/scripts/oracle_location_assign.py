"""동맥류 위치라벨(29클래스) 규칙기반 할당 — 오라클 상한 측정.

질문:
  "stage1 동맥류 seg → 가장 가까운 vessel skeleton의 클래스 = 그 동맥류의 클래스"
  이 규칙이 GT 입력(완벽한 동맥류 마스크 + 완벽한 vessel 마스크)에서 몇 %를 맞추나?
  여기서 안 나오면 고칠 것은 모델이 아니라 규칙이다.

입력: Dataset510(동맥류, 29 위치클래스) GT + Dataset600(vessel, 36클래스) GT
출력:
  analysis/lesion_vessel_features.json   병변별 raw feature (재분석용 캐시)
  analysis/oracle_location_report.md     사람이 읽는 리포트

핵심 확인 3가지:
  1. vessel GT가 동맥류 sac을 포함하는가? (포함하면 neck 기반 교집합 방식이 자기참조가 됨)
  2. 위치클래스별로 "최근접 vessel class"가 일관적인가? (=규칙이 성립할 여지)
  3. 어떤 signature가 29클래스를 얼마나 분리하는가? (leave-one-out 정확도)
"""
import json, sys, time, collections
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

BASE = Path("/home/user/TopAneu/seg/sblee/nnunet")
RAW = BASE / "nnUNet_raw"
LOC_DIR = RAW / "Dataset510_TopAneuLoc"
VES_DIR = RAW / "Dataset600_TopAneuVessel"
OUT = BASE / "analysis"
FEAT_JSON = OUT / "lesion_vessel_features.json"
REPORT = OUT / "oracle_location_report.md"

MAX_R = 10.0          # mm, 이 반경 밖의 vessel class는 기록 안 함
RADII = [1.0, 2.0, 3.0, 5.0]   # partner set 정의에 쓸 반경 후보


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

        # 26-connectivity (공식 eval의 Betti-0 관례와 동일)
        lesions, n = ndimage.label(loc > 0, structure=np.ones((3, 3, 3)))
        for lid in range(1, n + 1):
            m = lesions == lid
            nvox = int(m.sum())
            vals, cnts = np.unique(loc[m], return_counts=True)
            gt_id = int(vals[np.argmax(cnts)])

            # 병변 bbox + MAX_R 마진으로 crop
            idx = np.argwhere(m)
            lo = idx.min(0); hi = idx.max(0) + 1
            marg = np.ceil(MAX_R / spacing).astype(int) + 1
            lo2 = np.maximum(lo - marg, 0); hi2 = np.minimum(hi + marg, np.array(loc.shape))
            sl = tuple(slice(a, b) for a, b in zip(lo2, hi2))
            mc, vc = m[sl], ves[sl]

            # 1) sac 포함 여부: 병변 voxel 위의 vessel 라벨 분포
            inside = vc[mc]
            ov = collections.Counter(int(x) for x in inside if x > 0)
            overlap = {ves_names[k]: int(v) for k, v in ov.items()}
            inside_frac = float((inside > 0).sum()) / max(nvox, 1)

            # 2) 병변 표면으로부터 각 vessel class까지의 최단거리(mm)
            dist = ndimage.distance_transform_edt(~mc, sampling=spacing)
            dmap = {}
            for c in np.unique(vc):
                if c == 0:
                    continue
                sel = (vc == c) & (~mc)
                if not sel.any():
                    dmap[ves_names[int(c)]] = 0.0     # 병변 내부에만 존재
                    continue
                d = float(dist[sel].min())
                if d <= MAX_R:
                    dmap[ves_names[int(c)]] = round(d, 3)

            # 3) centroid 기준(=원안) 최근접 클래스
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
        if i % 10 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}  lesions={len(rows)}  {time.time()-t0:.0f}s", flush=True)

    OUT.mkdir(exist_ok=True)
    json.dump({"lesions": rows, "skipped": skipped}, open(FEAT_JSON, "w"), indent=1, ensure_ascii=False)
    print(f"[extract] 저장 {FEAT_JSON}  병변 {len(rows)}개, skip {len(skipped)}건", flush=True)
    return rows, skipped


def nearest_class(r):
    """병변 표면 기준 최근접 vessel class (거리 동률이면 겹침이 큰 쪽)."""
    if not r["dist_mm"]:
        return None
    md = min(r["dist_mm"].values())
    tied = [c for c, d in r["dist_mm"].items() if d <= md + 1e-6]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda c: r["overlap"].get(c, 0))


def inside_major(r):
    """sac 내부를 칠하고 있는 vessel class 중 최빈 = 어노테이터가 암묵적으로 지정한 모혈관.
    미포함 병변은 최근접으로 backoff."""
    if r["overlap"]:
        return max(r["overlap"], key=r["overlap"].get)
    return nearest_class(r)


def partners(r, R, exclude):
    return frozenset(c for c, d in r["dist_mm"].items() if d <= R and c != exclude)


def loo_accuracy(rows, sig_fn):
    """leave-one-out: 나머지로 signature→위치클래스 룩업을 만들고 자기 자신을 예측.
    미본 signature는 (최근접 클래스만)으로 backoff, 그것도 없으면 최빈 클래스."""
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
    A("# 동맥류 위치할당 규칙 — 오라클 상한 (GT 동맥류 + GT vessel)\n")
    A(f"> 병변 {n}개 / {len(set(r['case'] for r in rows))}케이스. "
      f"생성: `scripts/oracle_location_assign.py`. 캐시: `analysis/lesion_vessel_features.json`\n")
    if skipped:
        A(f"> ⚠️ 지오메트리 불일치로 제외: {len(skipped)}건 — {skipped[:5]}\n")

    # --- 1. sac 포함 여부 ---
    fr = np.array([r["inside_vessel_frac"] for r in rows])
    A("\n## 1. vessel GT가 동맥류 sac을 포함하는가\n")
    A(f"- 병변 voxel 중 vessel 라벨이 칠해진 비율: 중앙값 **{np.median(fr):.3f}**, "
      f"평균 {fr.mean():.3f} (min {fr.min():.3f} / max {fr.max():.3f})")
    A(f"- 거의 전부 포함(≥0.9)인 병변: **{int((fr>=.9).sum())}/{n}** "
      f"({100*(fr>=.9).mean():.0f}%)")
    A(f"- 거의 미포함(≤0.1): **{int((fr<=.1).sum())}/{n}** ({100*(fr<=.1).mean():.0f}%)")
    A("\n**해석**: 포함률이 높으면 vessel GT가 동맥류까지 혈관으로 칠한 것 → "
      "\"병변 dilate ∩ vessel\"은 자기참조가 되어 neck 판정에 못 쓴다. "
      "이때는 병변 *바깥* 껍질(shell)만 보거나, 거리 기반(최근접)으로 가야 한다.")

    # --- 2. 위치클래스별 최근접 vessel class 일관성 ---
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
        s = ", ".join(f"{k}×{v}" for k, v in top)
        A(f"| {loc} ({len(rs)}) | {s} | {share:.0%} |")
    w = sum(p * k for _, k, p in purity) / n
    A(f"\n- **가중 평균 최빈비율(=최근접 클래스만으로 낼 수 있는 상한 근사): {w:.1%}**")

    # --- 2b. sac 내부 최빈 클래스로 같은 표 ---
    A("\n### 2b. sac 내부 최빈 vessel class 기준 (어노테이터가 칠한 소속)\n")
    A("| 위치클래스 (n) | sac 내부 최빈 vessel class 분포 | 최빈 비율 |")
    A("|---|---|---|")
    purity2 = []
    for loc, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        c = collections.Counter(inside_major(r) for r in rs)
        top = c.most_common(3)
        share = top[0][1] / len(rs)
        purity2.append((loc, len(rs), share))
        A(f"| {loc} ({len(rs)}) | {', '.join(f'{k}×{v}' for k, v in top)} | {share:.0%} |")
    w2 = sum(p * k for _, k, p in purity2) / n
    A(f"\n- **가중 평균 최빈비율: {w2:.1%}**  (최근접 기준 {w:.1%} 대비 {w2-w:+.1%}p)")

    # --- 3. signature별 leave-one-out 정확도 ---
    A("\n## 3. 규칙 변형별 29클래스 정확도 (leave-one-out)\n")
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
    A(f"\n- 최고: **{best[0]} → {best[1]:.1%}**")
    A(f"- 참고: 최빈 클래스만 찍는 baseline = "
      f"{max(collections.Counter(r['gt_loc'] for r in rows).values())/n:.1%}")

    # --- 4. 거리 분포 (FP 필터 가능성) ---
    d0 = np.array([min(r["dist_mm"].values()) if r["dist_mm"] else 99 for r in rows])
    A("\n## 4. 병변–혈관 거리 (FP 필터용)\n")
    A(f"- 최근접 혈관까지 거리: 중앙값 {np.median(d0):.2f}mm, "
      f"95퍼센타일 {np.percentile(d0,95):.2f}mm, 최대 {d0.max():.2f}mm")
    A("- 진짜 동맥류가 이 범위 안에 있다는 뜻 → 예측 병변이 이보다 훨씬 멀면 FP로 버릴 수 있다.")

    REPORT.write_text("\n".join(L) + "\n")
    print(f"[report] 저장 {REPORT}", flush=True)
    print("\n".join(L[-14:]), flush=True)


if __name__ == "__main__":
    if "--reuse" in sys.argv and FEAT_JSON.exists():
        d = json.load(open(FEAT_JSON))
        rows, skipped = d["lesions"], d["skipped"]
        print(f"[cache] {FEAT_JSON} 재사용, 병변 {len(rows)}개", flush=True)
    else:
        rows, skipped = extract()
    report(rows, skipped)
