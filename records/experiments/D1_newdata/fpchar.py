#!/usr/bin/env python3
"""FPCHAR — 우리가 실제로 만드는 FP 의 해부 (2026-09-17 · 사용자 지시).

왜 다시 재나
    오늘 오전 `mvsweep.py` 는 **blob 개수와 GT 적중**만 셌다. 공식 지표의 FP 단위는
    (케이스 × 클래스) 이고, **gC 는 blob 하나에 라벨을 둘 낸다**. 그래서 blob 1개가
    FP 단위를 2개 만들 수 있고, 반대로 blob 을 1개 지워도 FP 단위가 2개 줄 수 있다.
    "test blob 94→93 이라 효과 없다" 는 내 결론은 **틀린 단위로 잰 것**이다.
    동료 보고서(3폴드 검출기)가 min_vox 12 에서 FP 50→44 를 봤다 — 단위가 다르다.

무엇을 재나 (기준 `b1fg_pf` = 제출본구성 + FRAC 0.35 + OUT_GROW 1.32 · 시드 0~4)
    각 (케이스 × 예측클래스) 단위마다:
      · TP 인가 FP 인가 (GT 에 그 클래스가 있나)
      · 그 클래스를 만든 **검출 blob 의 크기** (복셀 · mm^3)
      · 그 클래스가 blob 안에서 **1등인가 2등(gC 조각)인가** (blob 내 복셀 점유 최대 = 1등)
      · GT 병변까지 최단거리 (mm) · 케이스에 GT 가 있나
      · 그 클래스의 학습표 병변 수
    그리고 핵심 표: **blob 크기 문턱 k 를 걸면 FP 단위와 TP 단위가 각각 몇 개 사라지나.**
    (오전에는 이걸 blob 수로만 봤다. 이번엔 지표가 실제로 세는 단위로 본다.)

판정에 쓰지 않는다 — 이건 진단이다. 규칙 고정이 필요한 실험은 이 결과를 보고 따로 연다.
"""
import os, sys, json, collections
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
H = f"{R}/experiments/H1_patchfilter"
P = f"{R}/experiments/_c1_realpred"
G = f"{R}/dataset/TopAneu/location_masks"
A = f"{R}/code/sblee/nnunet/analysis"
ST = np.ones((3, 3, 3), bool)
TAG = "b1fg_pf"
SEEDS = [0, 1, 2, 3, 4]
KS = [5, 8, 10, 12, 15, 20, 25, 30]          # 복셀 문턱
MM3 = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]          # mm^3 문턱 (동료 지적 — 모달리티마다 복셀 부피가 다르다)

splits = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]
trcnt = collections.Counter(r["gt_loc"] for r in json.load(open(f"{A}/e11_feat_hyb_ov.json")))
id2name = None


def _path(d, cid):
    short = cid.replace("topaneu_", "")
    for c in (f"{d}/{cid}.nii.gz", f"{d}/{short}.nii.gz", f"{d}/topaneu_{short}.nii.gz"):
        if os.path.exists(c):
            return c
    return None


def one(args):
    split, seed, cid = args
    pp = _path(f"{H}/pred/{TAG}_{split}_s{seed}", cid)
    gp = _path(G, cid)
    ap = _path(f"{P}/aneu_{split}_b1ff", cid)
    if pp is None or gp is None:
        return []
    im = nib.load(pp)
    pred = np.asanyarray(im.dataobj)
    sp = np.array(im.header.get_zooms()[:3], float)
    voxmm3 = float(np.prod(sp))
    gt = np.asanyarray(nib.load(gp).dataobj)
    aneu = (np.asanyarray(nib.load(ap).dataobj) > 0) if ap else np.zeros_like(pred, bool)

    gtcls = set(int(x) for x in np.unique(gt) if x)
    prcls = set(int(x) for x in np.unique(pred) if x)
    lab, nb = ndi.label(aneu, structure=ST)
    bsize = np.bincount(lab.ravel()) if nb else np.array([0])

    # GT 거리맵 (GT 없으면 None)
    gdist = None
    if gt.any():
        gdist = ndi.distance_transform_edt(gt == 0, sampling=sp)

    # blob 별 라벨 점유 — 1등/2등 판정
    rank = {}                                   # (blob, cls) -> 점유 복셀
    for b in range(1, nb + 1):
        mb = lab == b
        for c, n in zip(*np.unique(pred[mb], return_counts=True)):
            if c:
                rank[(b, int(c))] = int(n)
    best = {}
    for (b, c), n in rank.items():
        if b not in best or n > rank[(b, best[b])]:
            best[b] = c

    out = []
    for c in prcls:
        m = pred == c
        # 이 클래스를 만든 blob — 겹침이 가장 큰 것 (OUT_GROW 로 blob 밖까지 번질 수 있다)
        ov = np.unique(lab[m], return_counts=True)
        cand = [(b, n) for b, n in zip(*ov) if b]
        if cand:
            b = max(cand, key=lambda x: x[1])[0]
            bs = int(bsize[b])
            is1 = 1 if best.get(b) == c else 0
        else:
            b, bs, is1 = 0, 0, -1
        d = float(gdist[m].min()) if gdist is not None else float("nan")
        out.append(dict(split=split, seed=seed, case=cid, cls=int(c),
                        tp=int(c in gtcls), blob_vox=bs, blob_mm3=bs * voxmm3,
                        cls_vox=int(m.sum()), rank1=is1,
                        dist_gt=d, case_has_gt=int(bool(gtcls)),
                        voxmm3=voxmm3))
    return out


def main():
    jobs = [(sp, sd, c) for sp in ("test", "val") for sd in SEEDS for c in splits[sp]]
    print(f"[fpchar] {len(jobs)} 작업 · 태그 {TAG} · 시드 {SEEDS}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    json.dump(rows, open(f"{R}/experiments/D1_newdata/fpchar.json", "w"))
    print(f"[fpchar] 단위 {len(rows)}개 수집", flush=True)

    ns = len(SEEDS)
    O = []; W = O.append
    W("# FPCHAR — 우리가 실제로 만드는 FP 의 해부")
    W("")
    W(f"기준 `{TAG}` (= 제출본 구성 + FRAC 0.35 + OUT_GROW 1.32) · 시드 {SEEDS} · "
      "단위 = (케이스 × 예측클래스) · 아래 수치는 **시드평균**이다.")
    W("")
    W("> 오전 `mvsweep.py` 는 blob 개수를 셌다. 공식 지표는 (케이스×클래스) 를 센다. "
      "**gC 가 blob 하나에 라벨을 둘 내므로 두 단위는 다르다.** 이 문서는 지표가 세는 단위로 잰다.")
    W("")

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    for split in ("test", "val"):
        R_ = sel(split=split)
        fp = [r for r in R_ if not r["tp"]]
        tp = [r for r in R_ if r["tp"]]
        W(f"## {split} — 단위 총계 (시드평균)")
        W("")
        W(f"- TP 단위 **{len(tp)/ns:.1f}** · FP 단위 **{len(fp)/ns:.1f}**")
        W(f"- FP 중 **GT 있는 케이스**에서 나온 것 {sum(r['case_has_gt'] for r in fp)/ns:.1f} "
          f"({sum(r['case_has_gt'] for r in fp)/max(len(fp),1)*100:.0f}%)")
        f1 = [r for r in fp if r["rank1"] == 1]; f2 = [r for r in fp if r["rank1"] == 0]
        t1 = [r for r in tp if r["rank1"] == 1]; t2 = [r for r in tp if r["rank1"] == 0]
        W(f"- **FP 중 gC 2등 조각**: {len(f2)/ns:.1f} / {len(fp)/ns:.1f} "
          f"(**{len(f2)/max(len(fp),1)*100:.0f}%**) · 1등 {len(f1)/ns:.1f}")
        W(f"- 참고 TP 중 2등 조각: {len(t2)/ns:.1f} / {len(tp)/ns:.1f} "
          f"({len(t2)/max(len(tp),1)*100:.0f}%) · 1등 {len(t1)/ns:.1f}")
        if fp:
            d = np.array([r["dist_gt"] for r in fp if np.isfinite(r["dist_gt"])])
            if len(d):
                W(f"- FP 의 GT 까지 거리 중앙 **{np.median(d):.1f}mm** · 3mm 이내 "
                  f"{int((d<=3).sum())/ns:.1f}개")
        W("")
        W(f"### {split} — blob 크기 분포 (복셀)")
        W("")
        W("| | p10 | p25 | 중앙 | p75 |")
        W("|---|---|---|---|---|")
        for nm, S in (("FP", fp), ("TP", tp)):
            v = np.array([r["blob_vox"] for r in S if r["blob_vox"] > 0])
            if len(v):
                W(f"| {nm} | {np.percentile(v,10):.0f} | {np.percentile(v,25):.0f} | "
                  f"{np.median(v):.0f} | {np.percentile(v,75):.0f} |")
        W("")
        W(f"### {split} — ★ 크기 문턱 k 를 걸면 (지표가 세는 단위로)")
        W("")
        W("| k (복셀) | 지워지는 FP | 지워지는 TP | 손익 (FP−9×TP) |")
        W("|---|---|---|---|")
        for k in KS:
            df = sum(1 for r in fp if 0 < r["blob_vox"] < k) / ns
            dt = sum(1 for r in tp if 0 < r["blob_vox"] < k) / ns
            W(f"| {k} | **{df:.1f}** | {dt:.1f} | {df - 9*dt:+.1f} |")
        W("")
        W("TP:FP 가치비 9:1 (C60 실측 test TP +0.0110 / FP −0.0012). 손익이 양수여야 이득이다.")
        W("")
        W(f"### {split} — mm³ 문턱 (모달리티 보정)")
        W("")
        W("| k (mm³) | 지워지는 FP | 지워지는 TP | 손익 |")
        W("|---|---|---|---|")
        for k in MM3:
            df = sum(1 for r in fp if 0 < r["blob_mm3"] < k) / ns
            dt = sum(1 for r in tp if 0 < r["blob_mm3"] < k) / ns
            W(f"| {k} | **{df:.1f}** | {dt:.1f} | {df - 9*dt:+.1f} |")
        W("")

    # 클래스별 FP — 학습 표본과의 관계
    W("## 클래스별 FP (test+val · 시드평균 · 상위 15)")
    W("")
    W("| 클래스 | FP | TP | FP/TP | 학습 병변 | 2등조각 FP 비율 |")
    W("|---|---|---|---|---|---|")
    import d9xx_lib as L
    i2n, _ = L.official_location_names()
    byc = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        b = byc[r["cls"]]
        b[0 if r["tp"] else 1] += 1
        if not r["tp"] and r["rank1"] == 0:
            b[2] += 1
    for c, (t, f, f2) in sorted(byc.items(), key=lambda x: -x[1][1])[:15]:
        nm = i2n.get(c, str(c))
        tr = trcnt.get(nm, 0)
        ratio = "∞" if t == 0 else f"{f/t:.1f}"
        W(f"| {nm} | {f/ns:.1f} | {t/ns:.1f} | {ratio} | {tr} | {f2/max(f,1)*100:.0f}% |")
    open(f"{R}/experiments/V1_vessel_axis/RESULTS_FPCHAR.md", "w").write("\n".join(O) + "\n")
    print("\n".join(O), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts")
    main()
