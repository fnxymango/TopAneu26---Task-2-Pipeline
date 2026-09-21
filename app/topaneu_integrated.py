"""Integrated TopAneu Task-2 pipeline: sblee's front end + jslee's hallucination filter.

Design decisions, each backed by a measurement in repro_task2/COMPONENTS.md:

  * ONE front end, not two. Unioning sblee's and jslee's detectors reached 3 more GT classes on
    test, but every one of those was a single case in a class with 1-2 GT instances -- and sblee
    uniquely reached 3 equally rare classes the other missed. That is noise on rare classes, not
    a capability difference, and two front ends cost ~690-890 s on a T4 against a 420 s limit.
  * jslee's patch-CNN hallucination filter IS kept: +0.0148 measured here, matching their own
    +0.0178, and it costs ~2 s a case.
  * jskim contributes no component -- their front end added 7 TP-pairs but zero new classes.

The two code bases cannot share one interpreter cleanly: sblee's stages need the vendored
Skeleton-Recall nnU-Net fork on PYTHONPATH, while jslee's patch filter imports the pip-installed
nnunetv2 indirectly. They are therefore run as separate subprocesses, which is exactly how the
configuration was validated offline.

Environment knobs (all optional; defaults are the validated configuration):
  TOPANEU_DET_FOLDS   detector folds, default "0,1,2,3,4"  -- the runtime/accuracy dial
  TOPANEU_PATCHCLF    patch-classifier dir, default <model>/patchclf; unset dir = filter off
  TOPANEU_PF_THRESH   override the threshold baked into the classifier's meta.json
  TOPANEU_TIMING      "1" prints per-stage wall time (used for the resource sanity check)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

APP = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get("TOPANEU_BUNDLE", "/opt/app/topaneu"))
MODEL_ROOT = Path(os.environ.get("TOPANEU_MODEL_DIR", "/opt/ml/model"))
TIMING = os.environ.get("TOPANEU_TIMING", "1") == "1"


def _log(msg: str) -> None:
    print(f"[integrated] {msg}", flush=True)


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        if TIMING:
            _log(f"{self.name}: {time.time() - self.t:.1f}s")


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("nan")


def _run_sblee(img: sitk.Image, tmp: Path) -> Path:
    """sblee stages 1-8: normalization, detection, vessel, post-proc, branch points, RF, gC.

    Run as a subprocess so the vendored nnU-Net fork can shadow the installed nnunetv2 without
    affecting the parent interpreter. The bundle locates its own params relative to --bundle,
    so nothing here may rearrange that tree.
    """
    in_path, out_path = tmp / "in.nii.gz", tmp / "loc.nii.gz"
    sitk.WriteImage(img, str(in_path), True)
    env = dict(os.environ, PYTHONPATH=str(BUNDLE / "vendor" / "Skeleton-Recall"),
               TOPANEU_EXPORT_VESSEL=str(tmp / "vespp.nii.gz"))
    cmd = [sys.executable, str(BUNDLE / "code/sblee/nnunet/scripts/pipeline_case.py"),
           "--image", str(in_path), "--modality", os.environ["TOPANEU_MODALITY"],
           "--out", str(out_path), "--bundle", str(BUNDLE),
           "--device", "cuda" if _cuda() else "cpu"]
    r = subprocess.run(cmd, env=env, text=True)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"sblee pipeline failed (rc={r.returncode})")
    return out_path


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _apply_patch_filter(loc_path: Path, tmp: Path, img: sitk.Image) -> Path:
    """jslee's hallucination filter, as a subprocess without the vendor fork on the path."""
    pf_dir = Path(os.environ.get("TOPANEU_PATCHCLF", str(MODEL_ROOT / "patchclf")))
    if not (pf_dir / "meta.json").exists():
        _log(f"patch filter skipped (no {pf_dir}/meta.json)")
        return loc_path
    ves = tmp / "vespp.nii.gz"
    if not ves.exists():
        _log("patch filter skipped (vessel map not exported by the sblee stage)")
        return loc_path
    out = tmp / "loc_filtered.nii.gz"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, str(APP / "run_patch_filter.py"),
           "--pred", str(loc_path), "--vessel", str(ves), "--image", str(tmp / "in.nii.gz"),
           "--patchclf", str(pf_dir), "--out", str(out),
           "--modality", os.environ["TOPANEU_MODALITY"]]
    if os.environ.get("TOPANEU_PF_THRESH"):
        cmd += ["--threshold", os.environ["TOPANEU_PF_THRESH"]]
    r = subprocess.run(cmd, env=env, text=True)
    if r.returncode != 0 or not out.exists():
        _log(f"patch filter failed (rc={r.returncode}) -- keeping unfiltered output")
        return loc_path
    return out


def predict_location(img: sitk.Image) -> sitk.Image:
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="topaneu_") as td:
        tmp = Path(td)
        with _Phase("sblee front end + assigner"):
            loc = _run_sblee(img, tmp)
        with _Phase("jslee patch filter"):
            loc = _apply_patch_filter(loc, tmp, img)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(loc))).astype(np.uint8)

    res = sitk.GetImageFromArray(arr)
    res.CopyInformation(img)
    labels = sorted(int(v) for v in np.unique(arr) if v)
    _log(f"done in {time.time() - t0:.1f}s · peak RSS {_rss_gb():.2f} GB · labels {labels}")
    return res
