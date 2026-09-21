import sys, os, json, importlib.util, io, contextlib
import numpy as np, SimpleITK as sitk
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; G=f"{R}/experiments/G1_gc_neweval"
sp=sys.argv[1]; tagoff=sys.argv[2]; tagon=sys.argv[3]
spec=importlib.util.spec_from_file_location("ne", os.path.expanduser("~/TopAneu-26/eval/task2/evaluate.py"))
sys.path.insert(0, os.path.expanduser("~/TopAneu-26/eval/task2")); ne=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(ne)
ids=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"][sp]
GT=f"{R}/dataset/TopAneu/location_masks"
names={v:k for k,v in json.load(open(f"{R}/dataset/TopAneu/location_mapping.json"))["labels"].items()}
def agg(tag):
    res=[]
    with contextlib.redirect_stdout(io.StringIO()):
        for c in ids:
            p=f"{G}/pred/{tag}/{c}.nii.gz"
            if os.path.exists(p): res.append(ne.evaluation_function(sitk.ReadImage(p), f"{GT}/{c}.nii.gz", execute_in_docker=False))
        return ne.evaluation_aggregation(res)
off,on=agg(tagoff),agg(tagon)
f=lambda v: "  nan " if np.isnan(v) else f"{v:6.3f}"
print(f"[{sp}] {tagoff} → {tagon} · 카운트가 달라진 클래스만")
print(f"{'cls':>3} {'이름':24s} {'sup':>3}  {'off tp/fp/fn':>13} {'on tp/fp/fn':>13}   {'MCC off':>7} {'MCC on':>7}   {'P off':>6} {'P on':>6}  {'HD95 off':>8} {'HD95 on':>8}")
for i in range(1,53):
    ko=(off[f"TP_{i}"],off[f"FP_{i}"],off[f"FN_{i}"]); kn=(on[f"TP_{i}"],on[f"FP_{i}"],on[f"FN_{i}"])
    if ko==kn: continue
    print(f"{i:>3} {names.get(i,'?')[:24]:24s} {off[f'support_{i}']:>3}  {str(ko):>13} {str(kn):>13}   {f(off[f'MCC_{i}']):>7} {f(on[f'MCC_{i}']):>7}   {f(off[f'PRECISION_{i}']):>6} {f(on[f'PRECISION_{i}']):>6}  {f(off[f'HD95_{i}']):>8} {f(on[f'HD95_{i}']):>8}")
