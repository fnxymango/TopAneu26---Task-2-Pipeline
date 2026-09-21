#!/usr/bin/env python3
"""
Modality-aware foreground intensity statistics for the TopAneu dataset.

목적
----
CTA(center4) / MRA(center2)를 하나의 modality-agnostic 데이터셋으로 합치기 전에,
"어느 값으로 clip 할지"를 교과서값이 아니라 **데이터의 전경(혈관/뇌) 분포**에서 직접 정한다.

산출물 (모두 sblee/outputs/intensity_stats 하위)
    - clip_recommendations.json   : 모달리티별 추천 clip 값 (CTA=글로벌 고정, MRA=per-volume 통계)
    - per_case_percentiles.csv    : 케이스별 percentile (재현/디버깅용)
    - hist_raw_vessel.png         : raw 스케일 혈관 전경 히스토그램 (CTA vs MRA) → 도메인 갭 확인
    - hist_normalized_vessel.png  : 추천 clip+[0,1] 적용 후 혈관 히스토그램 → modality 정렬 확인
    - spread_p995.png             : per-volume p99.5 분포 (MRA per-volume 필요성 근거)

주의
    - clip 상수는 최종적으로 **train fold에서만** 재계산할 것 (여기선 전체 98케이스로 계산 + 경고).
    - 전경 정의: brain = (image != 0), vessel = (vessel_mask > 0), aneurysm = (location_mask > 0).
      배경이 0으로 크롭돼 있어 image!=0 을 brain 전경 프록시로 사용.
"""
import os
import glob
import json
import csv
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 설정 ----
LOW_P, HIGH_P = 0.5, 99.5          # clip 경계로 쓸 percentile
EXTRA_PS = [1.0, 5.0, 95.0, 99.0]  # 참고용 추가 percentile
MAX_SAMPLES_PER_VOL = 20000         # 볼륨당 pooling 서브샘플 (히스토그램 메모리 관리)
RNG = np.random.default_rng(42)     # 재현성


def parse_modality(basename: str):
    """topaneu_center2_mr_002 -> ('MRA','center2'), topaneu_center4_ct_010 -> ('CTA','center4')"""
    parts = basename.split("_")
    center = parts[1]
    mod_token = parts[2].lower()
    modality = {"mr": "MRA", "ct": "CTA"}.get(mod_token, mod_token.upper())
    return modality, center


def foreground_mask(im: np.ndarray, modality: str) -> np.ndarray:
    """모달리티별 배경 제거.
    CTA: 공기(HU ~ -1000~-2048)를 -500 임계로 제거 → 연조직/혈관/뼈만.
    MRA: 배경이 0으로 크롭됨 → intensity > 0.
    """
    if modality == "CTA":
        return im > -500.0
    return im > 0.0


def subsample(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.size <= n:
        return arr
    idx = RNG.choice(arr.size, size=n, replace=False)
    return arr[idx]


def collect(dataset_root: str):
    img_dir = os.path.join(dataset_root, "images")
    ves_dir = os.path.join(dataset_root, "vessel_masks")
    loc_dir = os.path.join(dataset_root, "location_masks")
    images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    assert images, f"no images found in {img_dir}"

    per_case = []                              # dict per case
    pooled = {}                                # modality -> {'brain':[], 'vessel':[]}
    for i, imgp in enumerate(images, 1):
        base = os.path.basename(imgp).replace("_0000.nii.gz", "")
        modality, center = parse_modality(base)
        im = nib.load(imgp).get_fdata().astype(np.float32)
        vesp = os.path.join(ves_dir, f"{base}.nii.gz")
        locp = os.path.join(loc_dir, f"{base}.nii.gz")
        ves = nib.load(vesp).get_fdata() if os.path.exists(vesp) else np.zeros_like(im)
        loc = nib.load(locp).get_fdata() if os.path.exists(locp) else np.zeros_like(im)

        brain_vals = im[foreground_mask(im, modality)]
        vessel_vals = im[ves > 0]
        aneu_vals = im[loc > 0]

        def pct(a):
            if a.size == 0:
                return {}
            qs = [LOW_P, HIGH_P] + EXTRA_PS
            vals = np.percentile(a, qs)
            return {f"p{q}": float(v) for q, v in zip(qs, vals)}

        rec = {
            "case": base, "modality": modality, "center": center,
            "brain_n": int(brain_vals.size), "vessel_n": int(vessel_vals.size),
            "aneurysm_n": int(aneu_vals.size),
            "brain": pct(brain_vals), "vessel": pct(vessel_vals), "aneurysm": pct(aneu_vals),
        }
        per_case.append(rec)

        d = pooled.setdefault(modality, {"brain": [], "vessel": []})
        d["brain"].append(subsample(brain_vals, MAX_SAMPLES_PER_VOL))
        d["vessel"].append(subsample(vessel_vals, MAX_SAMPLES_PER_VOL))
        print(f"[{i:3d}/{len(images)}] {base:32s} {modality} "
              f"brain[{rec['brain'].get('p0.5','-'):.6}..{rec['brain'].get('p99.5','-'):.6}] "
              f"vessel_n={vessel_vals.size}", flush=True)

    for m in pooled:
        pooled[m]["brain"] = np.concatenate(pooled[m]["brain"]) if pooled[m]["brain"] else np.array([])
        pooled[m]["vessel"] = np.concatenate(pooled[m]["vessel"]) if pooled[m]["vessel"] else np.array([])
    return per_case, pooled


def recommend(per_case, pooled):
    rec = {"_note": "clip 상수는 최종적으로 train fold에서만 재계산할 것. 아래는 전체 98케이스 기준.",
           "percentiles_used": {"low": LOW_P, "high": HIGH_P}, "modalities": {}}
    for m, d in pooled.items():
        brain = d["brain"]
        glob_lo, glob_hi = np.percentile(brain, [LOW_P, HIGH_P])
        # per-volume p0.5/p99.5 분포 (per-volume 방식의 변동성)
        lows = [c["brain"].get(f"p{LOW_P}") for c in per_case if c["modality"] == m and c["brain"]]
        highs = [c["brain"].get(f"p{HIGH_P}") for c in per_case if c["modality"] == m and c["brain"]]
        lows, highs = np.array(lows), np.array(highs)
        ves_lo, ves_hi = (np.percentile(d["vessel"], [LOW_P, HIGH_P]) if d["vessel"].size else (None, None))

        # 추천 window: 상한은 혈관 밝기를 보존하도록 tissue/vessel 중 큰 값 사용.
        #   MRA는 혈관이 전경에서 제일 밝아(tissue p99.5 < vessel p99.5) → vessel 상한 채택.
        #   CTA는 뼈 때문에 tissue p99.5 > vessel p99.5 → tissue 상한 유지(혈관은 안쪽에 안전).
        win_hi = float(max(glob_hi, ves_hi)) if ves_hi is not None else float(glob_hi)
        win_lo = float(glob_lo)

        rec["modalities"][m] = {
            "n_cases": int(sum(1 for c in per_case if c["modality"] == m)),
            "global_tissue_clip": [round(float(glob_lo), 2), round(float(glob_hi), 2)],
            "vessel_fg_range": [None if ves_lo is None else round(float(ves_lo), 2),
                                None if ves_hi is None else round(float(ves_hi), 2)],
            "recommended_window": [round(win_lo, 2), round(win_hi, 2)],
            "per_volume_tissue_p_low":  {"min": float(lows.min()), "median": float(np.median(lows)), "max": float(lows.max())},
            "per_volume_tissue_p_high": {"min": float(highs.min()), "median": float(np.median(highs)), "max": float(highs.max())},
            "per_volume_p_high_cv": float(highs.std() / highs.mean()),  # 변동계수: 클수록 per-volume 필요
        }
    # 모달리티별 권고 방식
    rec["recommendation"] = {
        "CTA": "recommended_window를 글로벌 고정 상수로 사용 (HU=물리량). 혈관이 window 안쪽에 안전.",
        "MRA": "recommended_window 근방을 per-volume [p0.5, vessel-p99.5]로 재계산해 사용 "
               "(intensity 임의스케일 + 혈관이 최밝). 변동계수(cv) 확인.",
        "next": "clip+[0,1] 후 hist_normalized_vessel.png에서 두 모달리티 혈관 봉우리가 겹치는지 확인. "
                "어긋나면 vessel-peak landmark matching 추가.",
    }
    return rec


def apply_clip_scale(vals, lo, hi):
    v = np.clip(vals, lo, hi)
    return (v - lo) / (hi - lo + 1e-8)


def plot_raw_vessel(pooled, out):
    plt.figure(figsize=(8, 5))
    for m, d in pooled.items():
        if d["vessel"].size:
            plt.hist(d["vessel"], bins=200, density=True, alpha=0.5, label=f"{m} (n_vox={d['vessel'].size})")
    plt.xlabel("raw intensity"); plt.ylabel("density")
    plt.title("Vessel-foreground intensity (RAW): modality domain gap")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_normalized_vessel(pooled, rec, out):
    plt.figure(figsize=(8, 5))
    for m, d in pooled.items():
        if not d["vessel"].size:
            continue
        lo, hi = rec["modalities"][m]["recommended_window"]
        norm = apply_clip_scale(d["vessel"], lo, hi)
        plt.hist(norm, bins=200, density=True, alpha=0.5,
                 label=f"{m} clip[{lo:.0f},{hi:.0f}]->[0,1]")
    plt.xlabel("normalized [0,1]"); plt.ylabel("density")
    plt.title("Vessel-foreground after clip+[0,1]: modality alignment check")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_spread(per_case, out):
    mods = sorted({c["modality"] for c in per_case})
    data = [[c["brain"].get(f"p{HIGH_P}") for c in per_case
             if c["modality"] == m and c["brain"]] for m in mods]
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, tick_labels=mods, showfliers=True)
    plt.ylabel(f"per-volume tissue p{HIGH_P}")
    plt.title("Per-volume p99.5 spread (justifies MRA per-volume clip)")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/home/user/dataset/TopAneu")
    ap.add_argument("--out_dir",
                    default="/home/user/TopAneu/seg/sblee/outputs/intensity_stats")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_case, pooled = collect(args.dataset_root)
    rec = recommend(per_case, pooled)

    # JSON
    with open(os.path.join(args.out_dir, "clip_recommendations.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    # CSV (case별 주요 percentile)
    with open(os.path.join(args.out_dir, "per_case_percentiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "modality", "center", "brain_n", "vessel_n", "aneurysm_n",
                    "brain_p0.5", "brain_p99.5", "vessel_p0.5", "vessel_p99.5"])
        for c in per_case:
            w.writerow([c["case"], c["modality"], c["center"], c["brain_n"], c["vessel_n"],
                        c["aneurysm_n"],
                        c["brain"].get("p0.5"), c["brain"].get("p99.5"),
                        c["vessel"].get("p0.5"), c["vessel"].get("p99.5")])

    # plots
    plot_raw_vessel(pooled, os.path.join(args.out_dir, "hist_raw_vessel.png"))
    plot_normalized_vessel(pooled, rec, os.path.join(args.out_dir, "hist_normalized_vessel.png"))
    plot_spread(per_case, os.path.join(args.out_dir, "spread_p995.png"))

    print("\n===== 추천 clip =====")
    print(json.dumps(rec["modalities"], indent=2, ensure_ascii=False))
    print("\nsaved ->", args.out_dir)


if __name__ == "__main__":
    main()
