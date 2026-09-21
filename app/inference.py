"""GC entry point for the integrated Task-2 model.

main.py (unmodified challenge template) calls infer_ct / infer_mr with a SimpleITK image and
expects a uint8 SimpleITK image whose geometry matches the input exactly.

Modality comes from which interface Grand Challenge invoked -- never from the filename. The old
filename-sniffing path is what broke sblee's first container, where inputs arrive as anonymised
UUIDs.
"""
import os
from pathlib import Path

os.environ.setdefault("nnUNet_compile", "f")
os.environ.setdefault("nnUNet_raw", "/tmp/nnunet/raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet/preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnunet/results")

import SimpleITK as sitk


def _run(img: sitk.Image, modality: str) -> sitk.Image:
    os.environ["TOPANEU_MODALITY"] = modality
    import topaneu_integrated as TI
    return TI.predict_location(img)


def infer_ct(img: sitk.Image) -> sitk.Image:
    return _run(img, "CT")


def infer_mr(img: sitk.Image) -> sitk.Image:
    return _run(img, "MR")
