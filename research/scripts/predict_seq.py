#!/usr/bin/env python3
"""멀티프로세싱 없는 순차 예측기.

nnUNetv2_predict CLI는 전처리/저장을 백그라운드 워커로 돌리는데,
이 데이터셋(36클래스·큰 볼륨)에서 워커가 죽거나(KeyError / workers no longer alive)
케이스 종료 후 export 대기에서 행(hang)이 걸린다. predict_single_npy_array는
전 과정을 메인 프로세스에서 수행하므로 그 실패 모드가 없다.

사용: predict_seq.py <fold> <part_id> <num_parts>   (GPU는 CUDA_VISIBLE_DEVICES로 지정)
이미 있는 결과는 건너뛴다(재실행 안전).
"""
import sys, os, json, time
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# 두 가지 호출 방식:
#   predict_seq.py <fold> <part_id> <num_parts>      — 인덱스 분할
#   predict_seq.py <fold> --cases <c1,c2,...>        — 케이스 명시(진행 중인 것과 충돌 회피용)
fold = int(sys.argv[1])
explicit = None
if sys.argv[2] == '--cases':
    explicit, part, nparts = sys.argv[3].split(','), 0, 1
else:
    part, nparts = int(sys.argv[2]), int(sys.argv[3])
B = '/home/user/TopAneu/seg/sblee/nnunet'
N = f'{B}/experiments/D600_vessel_skelrec_resencm_250ep_bd0'
MODEL = (f'{N}/results/Dataset600_TopAneuVessel/'
         'nnUNetTrainerSkeletonRecallNoMirroring_250ep__nnUNetResEncUNetMPlansBD0__3d_fullres')
IN, OUT = f'{N}/cv/fold{fold}/in', f'{N}/cv/fold{fold}/out'
os.makedirs(OUT, exist_ok=True)

cases = sorted(json.load(open(f'{B}/nnUNet_preprocessed/Dataset600_TopAneuVessel/'
                             'splits_final.json'))[fold]['val'])
mine = explicit if explicit else [c for i, c in enumerate(cases) if i % nparts == part]
todo = [c for c in mine if not os.path.exists(f'{OUT}/{c}.nii.gz')]
print(f"[fold{fold} part{part}/{nparts}] 담당 {len(mine)} · 남은 {len(todo)}", flush=True)
if not todo:
    sys.exit(0)

pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                       perform_everything_on_device=True, device=torch.device('cuda'),
                       verbose=False, verbose_preprocessing=False, allow_tqdm=False)
pred.initialize_from_trained_model_folder(MODEL, use_folds=(fold,),
                                          checkpoint_name='checkpoint_final.pth')
io = SimpleITKIO()
for i, cid in enumerate(todo, 1):
    t = time.time()
    img, props = io.read_images([f'{IN}/{cid}_0000.nii.gz'])
    seg = pred.predict_single_npy_array(img, props, None, None, False)
    io.write_seg(seg, f'{OUT}/{cid}.nii.gz', props)
    print(f"  [{i}/{len(todo)}] {cid}  {time.time()-t:.0f}s", flush=True)
print(f"[fold{fold} part{part}] 완료", flush=True)
