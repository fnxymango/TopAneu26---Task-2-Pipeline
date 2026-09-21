#!/usr/bin/env python3
"""
TopAneu -> nnU-Net v2 raw format converter.

Builds 4 datasets (CTA/MRA x region5/loc), from
/home/user/dataset/TopAneu (images + location_masks).

- images: symlinked (no 6GB copy)
- labels: remapped location_masks written fresh
    * region5 : location value -> major vessel territory 1..5
    * loc     : present location values -> consecutive global ids 1..K
                (SAME mapping across CTA/MRA for comparability)
Run with sbaneu python (needs numpy, nibabel).
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np
import nibabel as nib

SRC   = Path("/home/user/dataset/TopAneu")
RAW   = Path("/home/user/TopAneu/seg/sblee/nnunet/nnUNet_raw")
IMGDIR = SRC / "images"
LABDIR = SRC / "location_masks"
LOCMAP = json.load(open(SRC / "location_mapping.json"))["labels"]
INV    = {v: k for k, v in LOCMAP.items()}          # int -> name

# ---- region (major vessel territory) by location value range ----
REGION_NAMES = {1: "VB_posterior", 2: "PCA", 3: "ICA", 4: "ACA_Acom", 5: "MCA"}
def region_of(v):
    if v <= 0:  return 0
    if v <= 17: return 1        # 1.x  VA/PICA/BA/AICA/SCA/BA-tip
    if v <= 21: return 2        # 2.x  P1P2/P3P4
    if v <= 35: return 3        # 3.x  ICA
    if v <= 44: return 4        # 4.x  Acom/ACA
    return 5                    # 5.x  MCA

def modality(cid):  # cid like topaneu_center4_ct_003
    return "CTA" if "_ct_" in cid else "MRA"

def cases():
    for ip in sorted(IMGDIR.glob("*_0000.nii.gz")):
        cid = ip.name.replace("_0000.nii.gz", "")
        yield cid, ip, LABDIR / f"{cid}.nii.gz"

# ---- pass A: discover global set of present location values ----
print("[A] scanning location_masks for present labels ...")
present = set()
for cid, ip, lp in cases():
    arr = np.asanyarray(nib.load(str(lp)).dataobj)
    present.update(int(x) for x in np.unique(arr) if x != 0)
present = sorted(present)
loc_id = {v: i + 1 for i, v in enumerate(present)}     # value -> 1..K (global)
print(f"    present location values ({len(present)}): {present}")

# ---- dataset definitions ----
DATASETS = {
    "Dataset501_TopAneuCTAregion": dict(mod="CTA", scheme="region"),
    "Dataset502_TopAneuMRAregion": dict(mod="MRA", scheme="region"),
    "Dataset511_TopAneuCTAloc":    dict(mod="CTA", scheme="loc"),
    "Dataset512_TopAneuMRAloc":    dict(mod="MRA", scheme="loc"),
}
CH = {"CTA": "CT", "MRA": "MRA"}   # MRA name != "CT" -> nnU-Net z-score

def remap(arr, scheme):
    out = np.zeros_like(arr, dtype=np.uint8)
    if scheme == "region":
        for v in np.unique(arr):
            if v: out[arr == v] = region_of(int(v))
    else:
        for v in np.unique(arr):
            if v: out[arr == v] = loc_id[int(v)]
    return out

# ---- build ----
counts = {name: 0 for name in DATASETS}
for name, cfg in DATASETS.items():
    d = RAW / name
    if d.exists(): shutil.rmtree(d)
    (d / "imagesTr").mkdir(parents=True)
    (d / "labelsTr").mkdir(parents=True)

for cid, ip, lp in cases():
    mod = modality(cid)
    lab_img = nib.load(str(lp))
    arr = np.asanyarray(lab_img.dataobj).astype(np.int16)
    for name, cfg in DATASETS.items():
        if cfg["mod"] != mod: continue
        d = RAW / name
        # image symlink
        link = d / "imagesTr" / f"{cid}_0000.nii.gz"
        if link.exists() or link.is_symlink(): link.unlink()
        os.symlink(ip.resolve(), link)
        # remapped label
        new = remap(arr, cfg["scheme"])
        nib.save(nib.Nifti1Image(new, lab_img.affine, lab_img.header),
                 str(d / "labelsTr" / f"{cid}.nii.gz"))
        counts[name] += 1

# ---- dataset.json ----
for name, cfg in DATASETS.items():
    if cfg["scheme"] == "region":
        labels = {"background": 0, **{v: k for k, v in REGION_NAMES.items()}}
    else:
        labels = {"background": 0}
        for v in present:
            labels[INV[v]] = loc_id[v]
    dj = {
        "channel_names": {"0": CH[cfg["mod"]]},
        "labels": labels,
        "numTraining": counts[name],
        "file_ending": ".nii.gz",
    }
    json.dump(dj, open(RAW / name / "dataset.json", "w"), indent=2, ensure_ascii=False)
    print(f"[build] {name:32} n={counts[name]:3d}  labels={len(labels)}  ch={CH[cfg['mod']]}")

print("\nDONE. raw at", RAW)
