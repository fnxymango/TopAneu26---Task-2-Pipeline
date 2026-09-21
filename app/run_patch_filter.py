"""Apply jslee's patch-CNN hallucination filter to a finished location map.

Runs as its own process so it never sees the vendored nnU-Net fork that sblee's stages need on
PYTHONPATH. It only needs jslee's patch_filter and features modules, not their pipeline.

ch0 must be the INTENSITY-based brain-mask z-score, which is what the classifier was trained on
and is NOT the vessel-EDT normalization jslee's location model uses (inference.py:586-603).
Getting that wrong feeds it a distribution it has never seen and the failure is silent.
"""
import argparse, os, sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--vessel", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--patchclf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--modality", required=True, choices=["CT", "MR"])
ap.add_argument("--threshold", default="")
a = ap.parse_args()

from src import patch_filter as PF, features as FE

net, thr = PF.load(Path(a.patchclf))
if a.threshold:
    thr = float(a.threshold)

si = sitk.ReadImage(a.pred)
seg = sitk.GetArrayFromImage(si).astype(np.uint8)
if not seg.any():
    sitk.WriteImage(si, a.out, True)
    print("[pf] nothing to filter", flush=True)
    raise SystemExit(0)

ves = sitk.GetArrayFromImage(sitk.ReadImage(a.vessel)).astype(np.uint8)
img = sitk.GetArrayFromImage(sitk.ReadImage(a.image)).astype(np.float32)
sp = list(si.GetSpacing())[::-1]

bm, _ = FE.brain_mask_intensity(img, sp, a.modality == "CT")
zm = np.array(img.shape) / np.array(bm.shape)
bmask = ndi.zoom(bm.astype(np.uint8), zm, order=0).astype(bool)
if bmask.shape != img.shape:                      # zoom can land a voxel short
    b2 = np.zeros(img.shape, bool)
    sl = tuple(slice(0, min(x, y)) for x, y in zip(bmask.shape, img.shape))
    b2[sl] = bmask[sl]
    bmask = b2
img_norm = FE.zscore_in_mask(img, bmask)

dev = "cpu"
try:
    import torch
    if torch.cuda.is_available() and os.environ.get("TOPANEU_PATCHCLF_GPU", "1") != "0":
        dev = "cuda"
except ImportError:
    pass

seg2, nb, nd_drop, _ = PF.filter_blobs(seg, net, thr, img_norm, ves, sp, (0, 0, 0), device=dev)
o = sitk.GetImageFromArray(seg2.astype(np.uint8))
o.CopyInformation(si)
sitk.WriteImage(o, a.out, True)
print(f"[pf] threshold {thr:.6g} · {nd_drop}/{nb} blobs dropped", flush=True)
