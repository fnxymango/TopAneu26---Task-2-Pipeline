#!/usr/bin/env python3
"""새 공식 eval(TopAneu-26 origin/main 60765a5) 로 예측 디렉터리를 채점한다.
사용: neweval.py <pred_dir> <split> [라벨]
  출력: 6지표(class-avg, NaN 제외) + 각 지표의 valid 클래스 수. 비교용으로 구 eval 6지표도 함께.
"""
import sys, os, json, importlib.util
import numpy as np, SimpleITK as sitk
_real_stdout = sys.stdout
sys.stdout = sys.stderr   # evaluate.py 의 디버그 print 를 stderr 로
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
NEW = os.path.expanduser("~/TopAneu-26/eval/task2")
OLD = f"{R}/code/TopAneu-26/eval/task2"
pred_dir, split = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(pred_dir.rstrip("/"))
ids = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"][split]
GT = f"{R}/dataset/TopAneu/location_masks"
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path)); spec.loader.exec_module(m); sys.path.pop(0); return m
new = load(f"{NEW}/evaluate.py", "new_eval")
results = []
for cid in ids:
    pp = f"{pred_dir}/{cid}.nii.gz"
    if not os.path.exists(pp): print(f"  [skip] {cid} 예측 없음", file=sys.stderr); continue
    pred = sitk.ReadImage(pp)
    results.append(new.evaluation_function(pred, f"{GT}/{cid}.nii.gz", execute_in_docker=False))
agg = new.evaluation_aggregation(results)
avg = new.evaluation_average(agg)
out = {"label": label, "split": split, "n_cases": len(results), "new": avg}
# 구 eval (프로젝트 사본) — 비교용
try:
    old = load(f"{OLD}/evaluate.py", "old_eval")
    import nibabel as nib
    ores = []
    for cid in ids:
        pp = f"{pred_dir}/{cid}.nii.gz"
        if not os.path.exists(pp): continue
        g = np.asanyarray(nib.load(f"{GT}/{cid}.nii.gz").dataobj).astype(np.int16)
        old.load_gt = (lambda _g: (lambda fn: _g))(g)     # 구 eval 은 docker 경로에서 GT 를 읽으므로 주입
        ores.append(old.evaluation_function(np.asanyarray(nib.load(pp).dataobj).astype(np.int16), cid))
    oagg = old.evaluation_aggregation(ores); oavg = old.evaluation_average(oagg)
    out["old"] = oavg
except Exception as e:
    out["old_error"] = str(e)
sys.stdout = _real_stdout
print(json.dumps(out, indent=1, default=float))
