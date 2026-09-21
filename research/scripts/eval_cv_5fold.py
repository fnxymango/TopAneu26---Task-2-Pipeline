#!/usr/bin/env python3
"""5-fold 교차검증 집계 — 각 fold를 '그 fold가 안 본 자기 val'로만 채점(유출 없음).

held-out test 14는 건드리지 않는다. 84케이스(=98-14) 전체를 fold별로 한 번씩 커버.
케이스 단위로 프로세스 병렬 처리(skeletonize가 케이스당 수십 초라 직렬은 느림).

사용: eval_cv_5fold.py <라벨> <fold0_dir> <fold1_dir> ... <fold4_dir>
"""
import sys, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, '/home/user/TopAneu/seg/sblee/nnunet/scripts')
from postprocess_vessel import dice, cldice, NAME, NCLS, ST, GT, DS

NPROC = min(12, os.cpu_count() or 4)


def score_case(args):
    """한 케이스: 클래스별 Dice + clDice + 덩어리 수."""
    fold, d, cid = args
    g = sitk.GetArrayFromImage(sitk.ReadImage(f'{GT}/{cid}.nii.gz')).astype(np.int16)
    p = sitk.GetArrayFromImage(sitk.ReadImage(f'{d}/{cid}.nii.gz')).astype(np.int16)
    per = {}
    for c in range(1, NCLS + 1):
        gm = g == c
        if gm.sum() == 0:
            continue
        per[c] = dice(p == c, gm)
    return fold, cid, per, cldice(p > 0, g > 0), ndi.label(p > 0, structure=ST)[1]


if __name__ == '__main__':
    label = sys.argv[1]
    dirs = sys.argv[2:7]
    splits = json.load(open(f'{DS}/splits_final.json'))

    jobs = [(f, d, cid) for f, d in enumerate(dirs) for cid in splits[f]['val']]
    print(f"[{label}] {len(jobs)}케이스 채점 — 프로세스 {NPROC}개 병렬")
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        results = list(ex.map(score_case, jobs, chunksize=1))

    per_class = {c: [] for c in range(1, NCLS + 1)}       # 84케이스 전체 pooled
    alive_folds = {c: 0 for c in range(1, NCLS + 1)}      # 이 클래스를 예측한 fold 수
    by_fold = {f: {'dice': {c: [] for c in range(1, NCLS + 1)}, 'cl': [], 'ncc': []}
               for f in range(len(dirs))}
    for fold, cid, per, cl, ncc in results:
        for c, v in per.items():
            per_class[c].append(v); by_fold[fold]['dice'][c].append(v)
        by_fold[fold]['cl'].append(cl); by_fold[fold]['ncc'].append(ncc)

    fold_rows = []
    for f in range(len(dirs)):
        b = by_fold[f]
        fm = {c: float(np.mean(v)) for c, v in b['dice'].items() if v}
        for c, v in fm.items():
            if v > 0.01:
                alive_folds[c] += 1
        fold_rows.append({'fold': f, 'n_cases': len(b['cl']), 'n_classes': len(fm),
                          'mean_dice': float(np.mean(list(fm.values()))),
                          'mean_cldice': float(np.mean(b['cl'])),
                          'median_cc': int(np.median(b['ncc'])),
                          'dead': [NAME[c] for c, v in fm.items() if v == 0]})
        r = fold_rows[-1]
        print(f"  fold{f}: {r['n_cases']:>2}케이스  Dice {r['mean_dice']:.4f}  "
              f"clDice {r['mean_cldice']:.4f}  덩어리 {r['median_cc']:>3}  0점 {len(r['dead'])}개")

    pooled = {NAME[c]: round(float(np.mean(v)), 4) for c, v in per_class.items() if v}
    overall = float(np.mean(list(pooled.values())))
    cv_dice = float(np.mean([r['mean_dice'] for r in fold_rows]))
    cv_cl = float(np.mean([r['mean_cldice'] for r in fold_rows]))
    dead = [k for k, v in pooled.items() if v == 0]

    print(f"\n[{label}] 5-fold CV 종합 (84케이스, 유출 없음)")
    print(f"  fold평균 mean Dice   {cv_dice:.4f}  "
          f"(fold별 편차 ±{np.std([r['mean_dice'] for r in fold_rows]):.4f})")
    print(f"  84케이스 pooled Dice {overall:.4f}")
    print(f"  fold평균 clDice      {cv_cl:.4f}")
    print(f"  전 fold 0점 클래스   {len(dead)}개: {', '.join(dead) if dead else '없음'}")

    print(f"\n  {'class':<14}{'pooled Dice':>12}{'살린 fold':>11}")
    for c in range(1, NCLS + 1):
        n = NAME[c]
        if n in pooled:
            print(f"  {n:<14}{pooled[n]:>12.3f}{alive_folds[c]:>9}/5")

    json.dump({'label': label, 'cv_mean_dice': cv_dice, 'pooled_dice': overall,
               'cv_mean_cldice': cv_cl, 'per_class': pooled,
               'alive_folds': {NAME[c]: alive_folds[c] for c in range(1, NCLS + 1)},
               'folds': fold_rows},
              open(f'/home/user/TopAneu/seg/sblee/nnunet/experiments/'
                   f'D600_vessel_skelrec_resencm_250ep_bd0/cv_{label}.json', 'w'), indent=2)
