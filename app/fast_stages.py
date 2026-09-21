"""Drop-in faster replacements for the two stages that scan the whole volume for no reason.

Measured on the submitted container (378x426x622 = 100M voxels, real hardware):
    branch-point graph  31.2 s   (10% of 312.9 s)
    detection filter    17.8 s   ( 6%)
Both compute over the entire volume while needing only a sparse part of it, so both can be cut
with output that is provably identical -- no accuracy is traded.

1) det_filter. The original builds a full-volume EDT of the vessel complement and then reads
   only `dist[blob].min()`, comparing it to max_dist (1.0 mm). The exact distance is never used
   beyond that comparison, so the EDT only has to be right within max_dist of each blob. Each
   blob is cropped with a margin of ceil(max_dist / spacing) + 1 voxels, which is wide enough
   that any vessel voxel capable of being within max_dist is inside the crop.

2) extract_nodes. Lee thinning runs on `ves > 0`, which occupies ~0.3% of the volume. Cropping to
   the vessel bounding box with >=2 voxels of background margin leaves the skeleton unchanged --
   thinning is local and the margin supplies the same background context the full volume did --
   and the node coordinates are shifted back afterwards.

Both are verified against the originals on real cases by verify_fast_stages.py; they are only
worth shipping if that check is exact.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

ST = np.ones((3, 3, 3), dtype=bool)


def filter_case_fast(aneu, ves, spacing, min_vox=5, max_dist=1.0):
    """Same contract as det_filter.filter_case, without the full-volume EDT."""
    lab, n = ndi.label(aneu > 0, structure=ST)
    if n == 0:
        return np.zeros_like(lab, dtype=np.int16)
    vmask = ves > 0
    if not vmask.any():
        return np.zeros_like(lab, dtype=np.int16)

    spacing = np.asarray(spacing, dtype=float)
    pad = np.ceil(max_dist / spacing).astype(int) + 1
    objs = ndi.find_objects(lab)
    keep = []
    for l, sl in enumerate(objs, 1):
        if sl is None:
            continue
        sub = lab[sl] == l
        if sub.sum() < min_vox:
            continue
        big = tuple(slice(max(0, s.start - p), min(d, s.stop + p))
                    for s, p, d in zip(sl, pad, lab.shape))
        vsub = vmask[big]
        if not vsub.any():
            continue                      # no vessel within reach -> distance > max_dist
        d = ndi.distance_transform_edt(~vsub, sampling=spacing)
        blob = np.zeros(d.shape, bool)
        inner = tuple(slice(s.start - b.start, s.stop - b.start) for s, b in zip(sl, big))
        blob[inner] = sub
        if float(d[blob].min()) <= max_dist:
            keep.append(l)
    return np.isin(lab, keep).astype(np.int16)


def crop_to_vessel(ves, margin=2):
    """Bounding box of the vessel mask with background margin; returns (crop, offset)."""
    nz = np.argwhere(ves > 0)
    if not len(nz):
        return ves, np.zeros(3, dtype=int)
    lo = np.maximum(nz.min(0) - margin, 0)
    hi = np.minimum(nz.max(0) + 1 + margin, np.array(ves.shape))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return ves[sl], lo
