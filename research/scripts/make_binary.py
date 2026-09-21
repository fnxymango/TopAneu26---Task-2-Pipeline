#!/usr/bin/env python3
"""
Build Dataset520_TopAneuBinary: combined CTA+MRA (98), aneurysm vs background.
Derived by binarizing Dataset500's region labels (any foreground -> 1).
"""
import json, os, shutil
from pathlib import Path
import numpy as np, nibabel as nib

RAW = Path("/home/user/TopAneu/seg/sblee/nnunet/nnUNet_raw")
src = RAW / "Dataset500_TopAneuRegion"
d   = RAW / "Dataset520_TopAneuBinary"
if d.exists(): shutil.rmtree(d)
(d / "imagesTr").mkdir(parents=True)
(d / "labelsTr").mkdir(parents=True)

n = 0
for lab in sorted((src / "labelsTr").glob("*.nii.gz")):
    cid = lab.name[:-7]
    # image: symlink to the original resolved target
    tgt = os.path.realpath(src / "imagesTr" / f"{cid}_0000.nii.gz")
    os.symlink(tgt, d / "imagesTr" / f"{cid}_0000.nii.gz")
    # label: binarize
    im = nib.load(str(lab)); arr = np.asanyarray(im.dataobj)
    b = (arr > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(b, im.affine, im.header), str(d / "labelsTr" / lab.name))
    n += 1

json.dump({"channel_names": {"0": "angiography"},
           "labels": {"background": 0, "aneurysm": 1},
           "numTraining": n, "file_ending": ".nii.gz"},
          open(d / "dataset.json", "w"), indent=2)
print(f"Dataset520_TopAneuBinary  n={n}  labels=2 (bg, aneurysm)  ch=angiography(z-score)")
