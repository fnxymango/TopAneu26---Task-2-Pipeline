"""Read-only audit of cached FRAC 0.35 predictions; writes only this directory.

Lesion diagnostics follow intweak.py: 6-connected GT components >=3 voxels,
with a 3-iteration voxel dilation for spatial matching. This is not the official
case-by-class metric. Diameter is volume-equivalent spherical diameter.
"""
import collections
import concurrent.futures
import csv
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

R = Path('/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee')
E = R / 'experiments'
H = E / 'H1_patchfilter'
OUT = Path(__file__).resolve().parent
S = json.loads((R / 'dataset/TopAneu/dataset_split.json').read_text())
NAMES = {int(k): v for k, v in S['location_classes'].items()}
META = {x['case_id']: x for sp in ('test', 'val') for x in S['cases'][sp]}
SEEDS = list(range(10))
TAG = 'b1frac035_pf'


def code(name):
    b = re.sub(r'^[RL]-', '', name)
    if b.startswith('5.3'):
        return '5.3d' if 'Distal' in b else '5.3j'
    return b.split()[0]


def region(name):
    return {'1': '후순환', '2': '후순환', '3': 'ICA', '4': 'ACA/Acom', '5': 'MCA'}[code(name)[0]]


def subgroup(name):
    c = code(name)
    if c in ('1.1', '1.2', '1.3'): return 'VA/PICA'
    if c == '1.10': return 'BA tip'
    if c.startswith('1.'): return 'BA 몸통/분지'
    if c.startswith('2.'): return 'PCA'
    if c == '3.1': return 'ICA C1-C5'
    if c in ('3.2', '3.3'): return 'ICA C6'
    if c in ('3.4', '3.5', '3.6'): return 'ICA C7 Pcom/AChA/nonBranch'
    if c == '3.7': return 'ICA C7 terminus'
    if c == '4.1': return 'Acom'
    if c == '4.2': return 'A1'
    if c.startswith('4.'): return 'A2 이상'
    if c == '5.1': return 'M1 trunk'
    if c in ('5.2', '5.3j'): return 'M1 분기부'
    return 'Distal M2/M3'


ADJ = {frozenset(p) for p in [
    ('3.1','3.2'),('3.2','3.3'),('3.3','3.4'),('3.4','3.5'),('3.5','3.6'),('3.6','3.7'),
    ('3.7','5.1'),('3.7','4.2'),('5.1','5.2'),('5.2','5.3j'),('5.3j','5.3d'),('5.1','5.3j'),
    ('4.2','4.1'),('4.1','4.3'),('4.3','4.4'),('4.4','4.5'),('4.2','4.3'),
    ('2.1','2.2'),('1.10','2.1'),('1.1','1.3'),('1.3','1.2'),('1.1','1.5'),('1.5','1.4'),
    ('1.4','1.7'),('1.7','1.6'),('1.4','1.9'),('1.9','1.8'),('1.4','1.10')]}


def error_kind(gt, pred):
    if pred == 0: return '미할당'
    if gt == pred: return '정답'
    gn, pn = NAMES[gt], NAMES[pred]
    gc, pc = code(gn), code(pn)
    if gc == pc: return '좌우반전'
    gs = gn[:1] if gn[:2] in ('R-', 'L-') else ''
    ps = pn[:1] if pn[:2] in ('R-', 'L-') else ''
    if frozenset((gc, pc)) in ADJ:
        return '인접분절' if (not gs or not ps or gs == ps) else '인접분절+좌우반전'
    return '같은혈관군 원거리' if gc[0] == pc[0] else '다른혈관군'


def read(path, ref=None):
    img = nib.load(path)
    if ref is not None:
        assert img.shape == ref.shape, (path, img.shape, ref.shape)
        assert np.allclose(img.affine, ref.affine, atol=1e-4), str(path)
    # NIfTI arrays are Fortran-contiguous. Reverse all three axes consistently
    # for contiguous scans/labeling; voxel adjacency and volume are unchanged.
    return img, np.asanyarray(img.dataobj).transpose(2, 1, 0)


def local_prediction(arr, sl, mask):
    vals, counts = np.unique(arr[sl][mask], return_counts=True)
    hist = {int(v): int(n) for v, n in zip(vals, counts) if v}
    dominant = max(hist, key=hist.get) if hist else 0
    return {'dominant': dominant, 'labels': sorted(hist), 'voxels': hist}


def case_audit(task):
    split, case = task
    gi, gt = read(R / f'dataset/TopAneu/location_masks/{case}.nii.gz')
    _, det = read(E / f'_c1_realpred/aneu_{split}_b1ff/{case}.nii.gz', gi)
    gt_classes = set(int(x) for x in np.unique(gt[gt != 0]))
    lesion_rows, masks = [], []
    for cls in sorted(gt_classes):
        lab, n = ndi.label(gt == cls)
        for idx, bb in enumerate(ndi.find_objects(lab), 1):
            if bb is None: continue
            sl = tuple(slice(max(0, s.start-3), min(gt.shape[j], s.stop+3)) for j,s in enumerate(bb))
            m = lab[sl] == idx
            nvox = int(m.sum())
            if nvox < 3: continue
            dilated = ndi.binary_dilation(m, iterations=3)
            masks.append((sl, dilated))
            nm = NAMES[cls]
            lesion_rows.append(dict(
                split=split, case=case, component=idx, cls=cls, name=nm,
                code=code(nm), region=region(nm), subgroup=subgroup(nm),
                modality=META[case]['modality'], nvox=nvox,
                equivalent_diameter_mm=float(2*(3*nvox*np.prod(gi.header.get_zooms()[:3])/(4*np.pi))**(1/3)),
                detected=bool(np.any(det[sl][dilated] > 0)), predictions=[]))
        del lab
    del det
    case_rows, fp_rows = [], []
    # Many seeds produce byte-identical masks. Decode each distinct file once,
    # retaining only small per-lesion summaries rather than full image arrays.
    decoded = {}
    def summarize_mask(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in decoded:
            _, pred = read(path, gi)
            pc = set(int(x) for x in np.unique(pred[pred != 0]))
            local = [local_prediction(pred, sl, mask) for sl, mask in masks]
            decoded[digest] = (pc, local)
        return decoded[digest]
    for seed in SEEDS:
        pc, local = summarize_mask(H / f'pred/{TAG}_{split}_s{seed}/{case}.nii.gz')
        for row, saved in zip(lesion_rows, local):
            lp = dict(saved)
            lp.update(seed=seed, dominant_correct=lp['dominant'] == row['cls'],
                      any_correct=row['cls'] in lp['labels'], kind=error_kind(row['cls'], lp['dominant']))
            row['predictions'].append(lp)
        for cls in sorted(pc - gt_classes):
            near = any(cls in p['labels'] for row in lesion_rows for p in row['predictions'][-1:])
            fp_rows.append(dict(split=split, case=case, seed=seed, cls=cls, name=NAMES[cls],
                                region=region(NAMES[cls]), near_gt=near))
        case_rows.append(dict(split=split, case=case, seed=seed, gt=sorted(gt_classes), pred=sorted(pc),
                              tp=sorted(pc & gt_classes), fp=sorted(pc-gt_classes), fn=sorted(gt_classes-pc)))
    # Seed 3 is the RF seed used in the submission: isolate filter and FRAC effects.
    for tag, key in [('b1frac035','before_patchfilter_seed3'), ('b1on_pf','baseline_seed3')]:
        _, local = summarize_mask(H / f'pred/{tag}_{split}_s3/{case}.nii.gz')
        for row, saved in zip(lesion_rows, local):
            row[key] = dict(saved)
    return lesion_rows, case_rows, fp_rows


def main():
    tasks = [(sp,c) for sp in ('test','val') for c in S['splits'][sp]]
    lesions, cases, fps = [], [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for i, (ls, cs, fs) in enumerate(pool.map(case_audit, tasks), 1):
            lesions.extend(ls); cases.extend(cs); fps.extend(fs)
            if i % 10 == 0 or i == len(tasks): print(f'{i}/{len(tasks)} cases audited', flush=True)
    # Validate case/class counts against independently generated frozen official scores.
    scores = {}
    for sp in ('test','val'):
        scores[sp] = {}
        for tag in ('b1on_pf', TAG):
            xs = [json.loads((H/f'scores/{tag}_{sp}_s{s}.json').read_text()) for s in SEEDS]
            assert all(x['n_cases'] == len(S['splits'][sp]) for x in xs)
            assert len({x['eval_commit'] for x in xs}) == 1
            scores[sp][tag] = xs
        for seed in SEEDS:
            x = scores[sp][TAG][seed]
            cc = [c for c in cases if c['split'] == sp and c['seed'] == seed]
            for cls in NAMES:
                for metric, key in [('TP','tp'),('FP','fp'),('FN','fn')]:
                    actual = sum(cls in c[key] for c in cc)
                    expected = x['counts'][f'{metric}_{cls}']
                    assert actual == expected, (sp,seed,cls,metric,actual,expected)
            assert x['counts'] == scores[sp]['b1on_pf'][seed]['counts'], (sp,seed,'FRAC changed class counts')
    for row in lesions:
        row['n_dominant_correct'] = sum(p['dominant_correct'] for p in row['predictions'])
        row['n_any_correct'] = sum(p['any_correct'] for p in row['predictions'])
        row['n_unassigned'] = sum(not p['labels'] for p in row['predictions'])
    data = dict(tag=TAG, seeds=SEEDS, n_cases=len(tasks),
                method='6-connected GT components >=3 voxels; spatial tolerance: 3 iterations of binary dilation',
                lesions=lesions, cases=cases, fps=fps)
    (OUT/'audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (OUT/'official_scores.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2))
    fields = ['split','case','component','cls','name','region','subgroup','modality','equivalent_diameter_mm',
              'detected','n_dominant_correct','n_any_correct','n_unassigned','seed3_dominant','seed3_labels',
              'seed3_pre_filter_labels','dominant_predictions']
    with (OUT/'lesions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in lesions:
            row={k:r[k] for k in fields if k in r}
            row.update(seed3_dominant=NAMES.get(r['predictions'][3]['dominant'],'미할당'),
                       seed3_labels='; '.join(NAMES[c] for c in r['predictions'][3]['labels']),
                       seed3_pre_filter_labels='; '.join(NAMES[c] for c in r['before_patchfilter_seed3']['labels']),
                       dominant_predictions=json.dumps(dict(collections.Counter(NAMES.get(p['dominant'],'미할당') for p in r['predictions'])),ensure_ascii=False))
            w.writerow(row)
    print(f'Completed: {len(lesions)} lesions; {len(cases)} case-seed records. Official TP/FP/FN counts all matched.',flush=True)


if __name__ == '__main__':
    main()
