import sys, os, time, json
sys.argv=["x"]
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; B=os.path.expanduser("~/e9_bundle_experiment")
def main():
    sys.path.insert(0, os.path.join(B,"vendor","Skeleton-Recall"))
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    M=os.path.join(B,"models/detector/Dataset722_TopAneuPjh3cls417","nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans722iso04__3d_fullres")
    t0=time.time()
    p=nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                      perform_everything_on_device=True, device=torch.device("cuda"),
                      verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(M, use_folds=(5,6,7), checkpoint_name="checkpoint_best.pth")
    p.predict_from_files(f"{R}/experiments/H4_folds/smoke_in", f"{R}/experiments/H4_folds/smoke_out",
                         save_probabilities=False, overwrite=True,
                         num_processes_preprocessing=2, num_processes_segmentation_export=2)
    print(f"[smoke] 1케이스 3폴드 {time.time()-t0:.1f}s", flush=True)
if __name__=="__main__":
    main()
