#!/usr/bin/env python3
"""neweval2.py — 동결된 공식 eval 로 채점한다. neweval.py 의 후속.

neweval.py 와 다른 점 셋:
 1. **동결 사본**(`eval_frozen/`)을 쓴다. 2026-09-11 H6 채점 도중 `~/TopAneu-26` 이 갱신돼
    한 실험 안에서 지표 집합이 갈렸다(F1 유무). 채점 중 eval 이 바뀌면 안 된다.
 2. 결과에 **eval 커밋 해시**를 남긴다. 나중에 섞였는지 바로 확인된다.
 3. **per-class TP/FP/FN/TN 집계를 함께 저장**한다. 앞으로 공식이 지표를 또 추가해도
    (F1 이 그랬듯) 재채점 없이 계산할 수 있다.

여러 예측 디렉터리를 한 프로세스에서 처리할 수 있다 — GT 로드를 공유해 재채점을 줄인다.
사용: neweval2.py <split> <pred_dir>[ <pred_dir> ...]
"""
import sys, os, json, importlib.util, time
import numpy as np, SimpleITK as sitk

_real = sys.stdout
sys.stdout = sys.stderr                      # eval 의 디버그 print 를 stderr 로
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
FROZEN = f"{R}/experiments/D1_newdata/eval_frozen"
OLD = f"{R}/code/TopAneu-26/eval/task2"
OUT = f"{R}/experiments/H1_patchfilter/scores"

def _hash():
    for ln in open(f"{FROZEN}/FROZEN.md"):
        if ln.startswith("커밋:"):
            return ln.split("**")[1]
    return "unknown"

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path)); spec.loader.exec_module(m); sys.path.pop(0)
    return m

def main():
    split = sys.argv[1]; dirs = sys.argv[2:]
    ids = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"][split]
    GT = f"{R}/dataset/TopAneu/location_masks"
    new = load(f"{FROZEN}/evaluate.py", "frozen_eval")
    try:
        old = load(f"{OLD}/evaluate.py", "old_eval"); import nibabel as nib
    except Exception:
        old = None
    H = _hash()
    for pd in dirs:
        tag = os.path.basename(pd.rstrip("/"))
        t0 = time.time(); results = []
        for cid in ids:
            pp = f"{pd}/{cid}.nii.gz"
            if not os.path.exists(pp):
                print(f"  [skip] {cid}", file=sys.stderr); continue
            results.append(new.evaluation_function(sitk.ReadImage(pp), f"{GT}/{cid}.nii.gz",
                                                   execute_in_docker=False))
        if not results:
            print(f"★{tag}: 예측 없음", file=sys.stderr); continue
        agg = new.evaluation_aggregation(results)
        out = {"label": tag, "split": split, "n_cases": len(results),
               "eval_commit": H, "new": new.evaluation_average(agg)}
        # 앞으로 지표가 또 늘어도 재채점 없이 계산하도록 원자료를 남긴다
        out["counts"] = {k: agg[k] for k in agg
                         if k.startswith(("TP_", "FP_", "FN_", "TN_", "support_"))}
        if old is not None:
            try:
                ores = []
                for cid in ids:
                    pp = f"{pd}/{cid}.nii.gz"
                    if not os.path.exists(pp): continue
                    g = np.asanyarray(nib.load(f"{GT}/{cid}.nii.gz").dataobj).astype(np.int16)
                    old.load_gt = (lambda _g: (lambda fn: _g))(g)
                    ores.append(old.evaluation_function(
                        np.asanyarray(nib.load(pp).dataobj).astype(np.int16), cid))
                out["old"] = old.evaluation_average(old.evaluation_aggregation(ores))
            except Exception as e:
                out["old_error"] = str(e)
        tmp = f"{OUT}/.n2_{tag}.{os.getpid()}"
        with open(tmp, "w") as fh: json.dump(out, fh, indent=1, default=float)
        os.replace(tmp, f"{OUT}/{tag}.json")
        print(f"  ✔ {tag} {time.time()-t0:.0f}s", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
