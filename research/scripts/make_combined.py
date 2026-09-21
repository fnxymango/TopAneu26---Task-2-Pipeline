#!/usr/bin/env python3
"""
Build COMBINED (CTA+MRA) single-model datasets from the per-modality ones:
  Dataset500_TopAneuRegion  <- 501 (CTA region) + 502 (MRA region)   [98, labels 0..5]
  Dataset510_TopAneuLoc     <- 511 (CTA loc)    + 512 (MRA loc)      [98, labels 0..29]

Images: symlinked to originals. Labels: copied (already remapped, consistent ids).
Channel name 'angiography' (!= CT) -> nnU-Net z-score normalization (safe for CT & MR).
"""
import json, os, shutil
from pathlib import Path

RAW = Path("/home/user/TopAneu/seg/sblee/nnunet/nnUNet_raw")
SRC_IMG = Path("/home/user/dataset/TopAneu/images")

COMBINE = {
    "Dataset500_TopAneuRegion": ["Dataset501_TopAneuCTAregion", "Dataset502_TopAneuMRAregion"],
    "Dataset510_TopAneuLoc":    ["Dataset511_TopAneuCTAloc",    "Dataset512_TopAneuMRAloc"],
}

for name, srcs in COMBINE.items():
    d = RAW / name
    if d.exists(): shutil.rmtree(d)
    (d / "imagesTr").mkdir(parents=True)
    (d / "labelsTr").mkdir(parents=True)
    base_labels = json.load(open(RAW / srcs[0] / "dataset.json"))["labels"]
    n = 0
    for src in srcs:
        s = RAW / src
        for lab in sorted((s / "labelsTr").glob("*.nii.gz")):
            cid = lab.name[:-7]                       # strip ".nii.gz"
            img = (SRC_IMG / f"{cid}_0000.nii.gz").resolve()
            link = d / "imagesTr" / f"{cid}_0000.nii.gz"
            if link.exists() or link.is_symlink(): link.unlink()
            os.symlink(img, link)
            shutil.copy2(lab, d / "labelsTr" / lab.name)
            n += 1
    dj = {"channel_names": {"0": "angiography"},
          "labels": base_labels, "numTraining": n, "file_ending": ".nii.gz"}
    json.dump(dj, open(d / "dataset.json", "w"), indent=2, ensure_ascii=False)
    print(f"[combined] {name:28} n={n:3d} labels={len(base_labels)} ch=angiography(z-score)")

print("DONE")
