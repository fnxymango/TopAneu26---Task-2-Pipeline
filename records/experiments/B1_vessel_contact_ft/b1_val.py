#!/usr/bin/env python3
"""B1 관문 V — val 41 에서 세 팔(base · ctrl · contact)의 ICA 곁가지 앵커율 · 가짜 앵커 · 36클래스 Dice.

규칙은 PLAN.md 에 결과 보기 전 고정. 세 팔 모두 같은 경로:
원본 영상 → nnUNetPredictor(tile 0.5 · gaussian · 미러링 없음) → V5 후처리 → C4 그래프.
사용: b1_val.py predict <arm>  (CUDA_VISIBLE_DEVICES 지정) · b1_val.py post <arm> · b1_val.py report
"""
import json, os, sys, glob, collections
import numpy as np, nibabel as nib
from pathlib import Path

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
B = f"{R}/experiments/B1_vessel_contact_ft"
SC = f"{R}/code/sblee/nnunet/scripts"
TR = "nnUNetResEncUNetMPlans__3d_fullres"
MODEL = {
    "base": ("/home/sblee/e9_bundle_experiment/models/vessel/Dataset800_TopAneuVessel417/"
             "nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__" + TR, "checkpoint_best.pth"),
    "ctrl": (f"{B}/results/Dataset801_TopAneuVesselFT/nnUNetTrainerVesselFT_ctrl__{TR}", "checkpoint_final.pth"),
    "contact": (f"{B}/results/Dataset801_TopAneuVesselFT/nnUNetTrainerVesselFT_contact__{TR}", "checkpoint_final.pth"),
}
BR = ("Pcom", "AChA", "OA")


def val_ids():
    return sorted(json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]["val"])


def predict(arm):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    md, ck = MODEL[arm]
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                        device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(md, use_folds=(0,), checkpoint_name=ck)
    out = f"{B}/val/ves_{arm}"; os.makedirs(out, exist_ok=True)
    for i, cid in enumerate(val_ids(), 1):
        if os.path.exists(f"{out}/{cid}.nii.gz"):
            continue
        img, props = SimpleITKIO().read_images([f"{R}/dataset/TopAneu/images/{cid}_0000.nii.gz"])
        p.predict_single_npy_array(img, props, None, f"{out}/{cid}", False)
        torch.cuda.empty_cache()
        print(f"[{arm}] {i}/41 {cid}", flush=True)


def post(arm):
    sys.path.insert(0, SC); os.environ.setdefault("TOPANEU_ROOT", R)
    import postprocess_vessel as PV, c4_branchpoint_graph as C4
    PV.apply(f"{B}/val/ves_{arm}", f"{B}/val/vespp_{arm}")
    g = f"{B}/val/graph_{arm}"; os.makedirs(g, exist_ok=True)
    adj, names = C4.adjacency_table(), C4.vessel_names()
    for cid in val_ids():
        C4.process_case(Path(f"{B}/val/vespp_{arm}/{cid}.nii.gz"), g, adj, names, C4.SPUR_MM, C4.JUNCTION_R_MM)


def has_anchor(nodes, sd, b):
    return any({f"{sd}-ICA-C6-C7", f"{sd}-{b}"} <= set(n["classes"]) and n.get("valid", True) for n in nodes)


def nodes(d, cid):
    return json.load(open(f"{d}/{cid}.json")).get("nodes", [])


def anchors(gdir):
    hit = fake = n = 0; per = collections.Counter(); pern = collections.Counter(); agree = []
    for cid in val_ids():
        ref, pr = nodes(f"{R}/experiments/_c4_bpgraph/all_ref", cid), nodes(gdir, cid)
        for sd in "RL":
            for b in BR:
                r, q = has_anchor(ref, sd, b), has_anchor(pr, sd, b)
                if r:
                    n += 1; hit += q; pern[b] += 1; per[b] += q
                elif q:
                    fake += 1
    return dict(n=n, hit=hit, fake=fake, A=hit / n, per={b: f"{per[b]}/{pern[b]}" for b in BR})


def dice_all(arm):
    sys.path.insert(0, SC); os.environ.setdefault("TOPANEU_ROOT", R)
    vals = []; percls = collections.defaultdict(list)
    for cid in val_ids():
        g = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz").dataobj).astype(np.int16)
        p = np.asanyarray(nib.load(f"{B}/val/vespp_{arm}/{cid}.nii.gz").dataobj).astype(np.int16)
        assert g.shape == p.shape, (cid, g.shape, p.shape)
        for c in range(1, 37):
            gm = g == c
            if gm.any():
                pm = p == c
                d = 2.0 * (pm & gm).sum() / (pm.sum() + gm.sum())
                vals.append(d); percls[c].append(d)
    return float(np.mean(vals)), {c: float(np.mean(v)) for c, v in percls.items()}


def report():
    res = {}
    for arm in ("base", "ctrl", "contact"):
        a = anchors(f"{B}/val/graph_{arm}"); md, pc = dice_all(arm)
        res[arm] = dict(**a, dice=md, pc=pc)
    old = anchors(f"{R}/experiments/_c4_bpgraph/val_pred")
    b = res["base"]
    print("# B1 관문 V — val 41 · ICA 곁가지 앵커 · 36클래스 Dice (규칙: PLAN.md)\n")
    print(f"참고: 기존 val_pred 그래프 앵커 {old['hit']}/{old['n']} · 가짜 {old['fake']} (기준 재추론 {b['hit']}/{b['n']} · 가짜 {b['fake']})\n")
    print("| 팔 | 앵커율 A | 적중/단위 | 가짜 | Pcom | AChA | OA | 평균 Dice |\n|---|---|---|---|---|---|---|---|")
    for arm, x in res.items():
        print(f"| {arm} | {x['A']:.3f} | {x['hit']}/{x['n']} | {x['fake']} | {x['per']['Pcom']} | {x['per']['AChA']} | {x['per']['OA']} | {x['dice']:.4f} |")
    ok = {}
    for arm in ("ctrl", "contact"):
        x = res[arm]
        c1 = x["A"] - b["A"] >= 0.10; c2 = (x["fake"] - b["fake"]) <= (x["hit"] - b["hit"]); c3 = x["dice"] >= b["dice"] - 0.01
        ok[arm] = c1 and c2 and c3
        print(f"\n- {arm}: ΔA {x['A'] - b['A']:+.3f} ({'✓' if c1 else '✗'}) · Δ가짜 {x['fake'] - b['fake']:+d} vs Δ적중 {x['hit'] - b['hit']:+d} ({'✓' if c2 else '✗'}) · ΔDice {x['dice'] - b['dice']:+.4f} ({'✓' if c3 else '✗'}) → {'통과' if ok[arm] else '미달'}")
    print("\n참고 · 관심 클래스 Dice (base / ctrl / contact):")
    for c, nm in ((4, "R-ICA-C6-C7"), (6, "L-ICA-C6-C7"), (8, "R-Pcom"), (9, "L-Pcom"), (31, "R-AChA"), (32, "L-AChA"), (33, "R-OA"), (34, "L-OA")):
        print(f"  {nm}: " + " / ".join(f"{res[a]['pc'].get(c, float('nan')):.3f}" for a in ("base", "ctrl", "contact")))
    passed = [a for a in ok if ok[a]]
    pick = max(passed, key=lambda a: (res[a]["A"], a == "contact")) if passed else None
    print(f"\n**관문 V → {'통과 · 선택 ' + pick + ' → 관문 E(e2e) 로' if pick else '미달 · B1(이 형태) 닫음'}**")
    json.dump(dict(pick=pick, ok=ok, res={a: {k: v for k, v in x.items() if k != 'pc'} for a, x in res.items()}),
              open(f"{B}/b1_gate_v.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "predict":
        predict(sys.argv[2])
    elif cmd == "post":
        post(sys.argv[2])
    else:
        report()
