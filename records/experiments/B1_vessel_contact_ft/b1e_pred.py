#!/usr/bin/env python3
"""B1 관문 E 1단계 — 선택된 fine-tune 체크포인트로 혈관을 다시 예측한다 (test 83 · train 291 · val 은 관문 V 것 재사용).

경로·설정은 pipeline_case.py 와 같다: 원본 영상 → nnUNetPredictor(tile 0.5 · gaussian · 미러링 없음 · fold0).
사용: b1e_pred.py <split> <shard> <nshard>   (CUDA_VISIBLE_DEVICES 로 GPU 지정 · 팔은 b1_gate_v.json 의 pick)
"""
import json, os, sys

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
B = f"{R}/experiments/B1_vessel_contact_ft"
TR = "nnUNetResEncUNetMPlans__3d_fullres"


def model_dir():
    arm = json.load(open(f"{B}/b1_gate_v.json"))["pick"]
    assert arm in ("ctrl", "contact"), arm
    return f"{B}/results/Dataset801_TopAneuVesselFT/nnUNetTrainerVesselFT_{arm}__{TR}", arm


def main(split, shard, nshard):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    md, arm = model_dir()
    cs = sorted(json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"][split])[shard::nshard]
    out = f"{B}/e2e/ves_{split}"; os.makedirs(out, exist_ok=True)
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                        device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(md, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    for i, cid in enumerate(cs, 1):
        if os.path.exists(f"{out}/{cid}.nii.gz"):
            continue
        img, props = SimpleITKIO().read_images([f"{R}/dataset/TopAneu/images/{cid}_0000.nii.gz"])
        p.predict_single_npy_array(img, props, None, f"{out}/{cid}", False)
        torch.cuda.empty_cache()
        print(f"[{arm}/{split}/{shard}] {i}/{len(cs)} {cid}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
