#!/usr/bin/env python3
"""작업 큐 방식 예측 워커 — GPU가 노는 것을 막는다.

fold별로 프로세스를 고정 배정하면 먼저 끝난 GPU가 놀게 된다(실제로 두 번 발생).
이 워커는 남은 케이스를 공용 큐에서 하나씩 '선점'해 처리하므로,
빈 GPU가 즉시 다음 케이스를 가져간다. 선점은 O_EXCL 파일 생성으로 원자적.

사용: CUDA_VISIBLE_DEVICES=n predict_queue.py [--reserve c1,c2,...]
      --reserve: 다른 프로세스가 이미 처리 중인 케이스(선점 처리해 중복 방지)
"""
import sys, os, json, time
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

B = '/home/user/TopAneu/seg/sblee/nnunet'
N = f'{B}/experiments/D600_vessel_skelrec_resencm_250ep_bd0'
MODEL = (f'{N}/results/Dataset600_TopAneuVessel/'
         'nnUNetTrainerSkeletonRecallNoMirroring_250ep__nnUNetResEncUNetMPlansBD0__3d_fullres')
CLAIMS = f'{N}/cv/claims'
os.makedirs(CLAIMS, exist_ok=True)
splits = json.load(open(f'{B}/nnUNet_preprocessed/Dataset600_TopAneuVessel/splits_final.json'))


def claim(fold, cid):
    """O_EXCL로 원자적 선점. 이미 있으면 다른 워커가 가져간 것."""
    try:
        fd = os.open(f'{CLAIMS}/f{fold}_{cid}', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


if '--reserve' in sys.argv:
    for item in sys.argv[sys.argv.index('--reserve') + 1].split(','):
        if not item:
            continue
        f, c = item.split(':', 1)
        claim(int(f), c)
    print(f"[reserve] 진행 중 케이스 선점 완료", flush=True)

# 남은 작업: fold 순으로 정렬(모델 재로딩 최소화)
todo = [(f, c) for f in range(1, 5) for c in sorted(splits[f]['val'])
        if not os.path.exists(f'{N}/cv/fold{f}/out/{c}.nii.gz')]
gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
print(f"[worker GPU{gpu}] 후보 {len(todo)}건", flush=True)

io = SimpleITKIO()
pred, loaded = None, None
n = 0
for f, cid in todo:
    if os.path.exists(f'{N}/cv/fold{f}/out/{cid}.nii.gz'):
        continue
    if not claim(f, cid):
        continue
    if loaded != f:
        pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                               perform_everything_on_device=True, device=torch.device('cuda'),
                               verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        pred.initialize_from_trained_model_folder(MODEL, use_folds=(f,),
                                                  checkpoint_name='checkpoint_final.pth')
        loaded = f
        print(f"[worker GPU{gpu}] fold{f} 모델 로드", flush=True)
    t = time.time()
    img, props = io.read_images([f'{N}/cv/fold{f}/in/{cid}_0000.nii.gz'])
    seg = pred.predict_single_npy_array(img, props, None, None, False)
    io.write_seg(seg, f'{N}/cv/fold{f}/out/{cid}.nii.gz', props)
    n += 1
    print(f"[worker GPU{gpu}] fold{f} {cid}  {time.time()-t:.0f}s  (누적 {n})", flush=True)
print(f"[worker GPU{gpu}] 큐 소진 — 종료 (처리 {n}건)", flush=True)
