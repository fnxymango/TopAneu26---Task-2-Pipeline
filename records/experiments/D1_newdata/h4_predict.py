#!/usr/bin/env python
"""h4_predict.py — E9 검출기를 지정한 폴드 부분집합으로 test/val 에 돌린다.

pipeline_case.py 의 _predict 와 **완전히 같은 설정**을 쓴다 (tile_step 0.5 · gaussian ·
mirroring off · checkpoint_best). 다른 게 있으면 폴드 수 비교가 아니라 설정 비교가 된다.

**본문은 반드시 main() 안에 둔다.** nnUNet 전처리 워커가 spawn 으로 이 모듈을 다시 import
하는데, 최상위에 실행 코드가 있으면 워커마다 전체가 재실행되어
"Background workers died / freeze_support()" 로 죽는다 (2026-09-10 실측).

사용: h4_predict.py --folds 5,6,7 --tag e9f3P5 --split test
"""
import argparse, json, os, sys, time

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
BUNDLE = os.path.expanduser("~/e9_bundle_experiment")
MODEL = os.path.join(BUNDLE, "models/detector/Dataset722_TopAneuPjh3cls417",
                     "nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans722iso04__3d_fullres")
RAW = f"{R}/nnunet/nnUNet_raw/Dataset722_TopAneuPjh3cls417/imagesTr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=["test", "val"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nproc", type=int, default=2)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(BUNDLE, "vendor", "Skeleton-Recall"))
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    OUTD = f"{R}/experiments/H4_folds/pred_{a.tag}_{a.split}"
    IND = f"{R}/experiments/H4_folds/in_{a.split}"
    ids = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"][a.split]
    os.makedirs(IND, exist_ok=True)
    os.makedirs(OUTD, exist_ok=True)
    for c in ids:                       # 이 split 만 담은 입력 폴더를 심볼릭으로 만든다
        src, dst = f"{RAW}/{c}_0000.nii.gz", f"{IND}/{c}_0000.nii.gz"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    have = sum(1 for c in ids if os.path.exists(f"{RAW}/{c}_0000.nii.gz"))
    print(f"[h4] {a.split} 입력 {have}/{len(ids)} · 폴드 {a.folds} · out {OUTD}", flush=True)
    if have != len(ids):
        print("★입력이 모자란다 — 정규화 영상 생성이 안 끝났다")
        return 1

    folds = tuple(int(x) for x in a.folds.split(","))
    t0 = time.time()
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        perform_everything_on_device=(a.device == "cuda"),
                        device=torch.device(a.device), verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(MODEL, use_folds=folds,
                                           checkpoint_name="checkpoint_best.pth")
    p.predict_from_files(IND, OUTD, save_probabilities=False, overwrite=False,
                         num_processes_preprocessing=a.nproc,
                         num_processes_segmentation_export=a.nproc)
    dt = time.time() - t0
    print(f"[h4] 완료 {len(ids)}케이스 · {dt:.0f}s · 케이스당 {dt/len(ids):.1f}s "
          f"· 폴드당 {dt/len(ids)/len(folds):.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
