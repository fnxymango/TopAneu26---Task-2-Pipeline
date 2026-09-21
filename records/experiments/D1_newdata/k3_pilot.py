#!/usr/bin/env python3
"""K3 사전 점검 — 혈관 모델 softmax 로 빠진 ICA 곁가지 분기점(Pcom·AChA·OA)을 되살릴 수 있나 (train 만 · GPU 추론).

C60 §30 "소분지 확률 문턱 — 보류(전볼륨 확률맵 케이스당 5.8GB)". 여기서는 확률을 디스크에 쓰지 않고 메모리에서
필요한 8채널(좌우 Pcom · AChA · OA · ICA-C6-C7)만 보고 버린다.
모델: 제출본과 같은 혈관 모델(Dataset800 · SkeletonRecall ClassWeightedV2 500ep · ResEncM · fold0 · checkpoint_best · 원본 영상).
대상: train 에서 ICA 원위(3.2~3.6) 병변이 있는 케이스.

측정(쪽 × 곁가지): 예측 혈관 그래프(_c4_bpgraph/vespp_train)에 ICA-C6-C7↔곁가지 노드가 **없는** 경우만 센다.
  참 결측 = 참조 그래프(all_ref · GT 혈관)에는 노드가 있음   /   원래 없음 = 참조에도 없음
  문턱 t 에서 '복원' = 곁가지 확률 ≥ t 인 연결성분(≥5복셀) 중 예측 ICA-C6-C7(vespp) 에서 2mm 안에 닿는 것이 있음
── 관문 (결과 보기 전 고정 · 2026-09-15) ──
 어떤 t ∈ {0.05, 0.1, 0.2, 0.3} 에서 [참 결측 복원률 ≥ 30%] ∧ [원래 없음 복원 수 ≤ 참 결측 복원 수] → K3 본 구축
 (415 케이스 확률 추론 · 그래프 보강 · 학습표/추론 그래프 재생성 · K0 장치 e2e). 아니면 K3 닫음.
 ⚠ fold0 은 train 으로 학습됐다 → train 확률은 in-sample(과대 가능). 통과해도 e2e 에서 다시 확인해야 한다.
사용: k3_pilot.py run <shard> <nshard>  (CUDA_VISIBLE_DEVICES 로 GPU 지정) · k3_pilot.py report
"""
import json, os, sys, re, collections
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
MODEL = "/home/sblee/e9_bundle_experiment/models/vessel/Dataset800_TopAneuVessel417/nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres"
OUT = f"{D}/k3_pilot"
CH = {"R": {"ICA": 4, "Pcom": 8, "AChA": 31, "OA": 33}, "L": {"ICA": 6, "Pcom": 9, "AChA": 32, "OA": 34}}
TS = (0.05, 0.1, 0.2, 0.3)
ST = np.ones((3, 3, 3), bool)


def has_anchor(nodes, sd, b):
    return any({f"{sd}-ICA-C6-C7", f"{sd}-{b}"} <= set(n["classes"]) and n.get("valid", True) for n in nodes)


def cases():
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    return sorted({r["case"] for r in rows if re.match(r"^[RL]-3\.[2-6] ", r["gt_loc"])})


def run(shard, nshard):
    import torch
    sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                        device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(MODEL, use_folds=(0,), checkpoint_name="checkpoint_best.pth")
    os.makedirs(OUT, exist_ok=True)
    cs = cases()[shard::nshard]
    for i, cid in enumerate(cs, 1):
        fo = f"{OUT}/{cid}.json"
        if os.path.exists(fo):
            continue
        img, props = SimpleITKIO().read_images([f"{R}/dataset/TopAneu/images/{cid}_0000.nii.gz"])
        _, prob = p.predict_single_npy_array(img, props, None, None, True)
        vi = nib.load(f"{R}/experiments/_c1_realpred/vespp_train/{cid}.nii.gz")
        ves = np.asanyarray(vi.dataobj); sp = np.array(vi.header.get_zooms()[:3], float)
        pr = {k: np.asarray(prob[k]).transpose(2, 1, 0) for s in CH.values() for k in s.values()}
        del prob
        assert pr[4].shape == ves.shape, (cid, pr[4].shape, ves.shape)
        g_pred = json.load(open(f"{R}/experiments/_c4_bpgraph/vespp_train/{cid}.json")).get("nodes", [])
        g_ref = json.load(open(f"{R}/experiments/_c4_bpgraph/all_ref/{cid}.json")).get("nodes", [])
        res = {}
        for sd, ch in CH.items():
            ica = ves == ch["ICA"]
            if ica.sum() < 30:
                continue
            idx = np.argwhere(ica); pad = np.ceil(15 / sp).astype(int)
            lo = np.maximum(idx.min(0) - pad, 0); hi = np.minimum(idx.max(0) + pad + 1, ves.shape)
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            dica = ndimage.distance_transform_edt(~ica[sl], sampling=sp)
            for b in ("Pcom", "AChA", "OA"):
                if has_anchor(g_pred, sd, b):
                    continue
                rec = {}
                for t in TS:
                    lab, n = ndimage.label(pr[ch[b]][sl] >= t, structure=ST)
                    ok = False
                    if n:
                        cnt = np.bincount(lab.ravel()); mind = ndimage.minimum(dica, lab, index=np.arange(1, n + 1))
                        ok = bool(np.any((cnt[1:] >= 5) & (np.asarray(mind) <= 2.0)))
                    rec[str(t)] = ok
                res[f"{sd}-{b}"] = dict(ref=has_anchor(g_ref, sd, b), rec=rec)
        json.dump(res, open(fo, "w"))
        print(f"[{shard}] {i}/{len(cs)} {cid} · 결측 {len(res)}", flush=True)


def report():
    cs = cases(); got = [c for c in cs if os.path.exists(f"{OUT}/{c}.json")]
    tot = collections.Counter()
    for c in got:
        for k, v in json.load(open(f"{OUT}/{c}.json")).items():
            b = k.split("-")[1]; kind = "참결측" if v["ref"] else "원래없음"
            tot[(kind, "n")] += 1; tot[(kind, b, "n")] += 1
            for t, ok in v["rec"].items():
                tot[(kind, t)] += ok; tot[(kind, b, t)] += ok
    print(f"# K3 사전 점검 — 혈관 softmax 로 빠진 ICA 곁가지 분기점 복원 (train · 케이스 {len(got)}/{len(cs)})\n")
    print(f"예측 그래프에서 빠진 (쪽×곁가지): 참 결측 {tot[('참결측','n')]} · 원래 없음 {tot[('원래없음','n')]}\n")
    print("| 문턱 | 참 결측 복원 | 복원률 | 원래 없음 복원(가짜) | 곁가지별 참결측 복원 (Pcom/AChA/OA) |\n|---|---|---|---|---|")
    best = None
    for t in map(str, TS):
        a, n = tot[("참결측", t)], tot[("참결측", "n")]; f = tot[("원래없음", t)]
        per = " / ".join(f"{tot[('참결측', b, t)]}/{tot[('참결측', b, 'n')]}" for b in ("Pcom", "AChA", "OA"))
        ok = n and a / n >= 0.30 and f <= a
        print(f"| {t} | {a}/{n} | {a/max(n,1):.0%} | {f}/{tot[('원래없음','n')]} | {per} | {'✓' if ok else ''}")
        if ok and best is None:
            best = t
    print(f"\n**관문(복원률 ≥30% ∧ 가짜 ≤ 복원) → {'통과 (t=' + best + ') · K3 본 구축' if best else '미달 · K3 닫음'}**")
    json.dump(dict(ok=best is not None, t=best, n_cases=len(got)), open(f"{D}/k3_pilot_gate.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        report()
