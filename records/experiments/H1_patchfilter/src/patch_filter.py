"""패치 CNN 환각 필터 (exp_55) 의 컨테이너 구현.

## 무엇을 하나

RF 재라벨링까지 끝난 blob 하나하나에서 **4채널 64^3 @0.5mm 패치**를 떼어 3D CNN 에 넣고,
'진짜 동맥류인가' 점수를 받는다. 임계 아래면 그 blob 을 지운다.

    ch0 영상(뇌마스크 z-score)   ch1 혈관 이진   ch2 혈관 36클래스/36   ch3 territory/5

시야 64 x 0.5 = **32mm** — 동맥류(3~10mm)와 모혈관 분기가 같이 들어간다.

## 임계값이 모델보다 중요하다 (exp_55 vs exp_56, val 42케이스 실측)

    같은 체크포인트(AUC 0.832)
      Youden J 임계 0.01235  -> official_all 0.2420 -> **0.2204**  (환각 18 제거, 검출 9 손실)
      보수 임계    0.000201  -> official_all 0.2420 -> **0.2598**  (환각  7 제거, 검출 0 손실)

공식 채점이 (케이스 x 클래스) 존재/부재라 **검출 손실 1(FN)과 환각 제거 1(FP)의 가중치가
같다.** Youden J(TPR-FPR)는 그 대칭을 안 본다. 임계는 학습 때 정해 meta.json 에 싣는다 —
추론 시점에 다시 고르지 않는다(고를 정답이 없다).

## 혈관은 stage-1 것을 그대로 쓴다

패치의 혈관 채널은 학습 때 **250ep farm**(D216 과 같은 farm)으로 만들었다. 컨테이너의
stage-1 vessel 이 정확히 그것이므로 추가 추론이 없다 — RF 피처(1250ep)와는 다르다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PATCH = 64
SPACING = 0.5
SPEC = "img+vbin+vseg36+terr/64@0.5mm/v1"
N_VES = 36


def load(model_dir: Path):
    """(net, threshold) 반환. 규격이 다르면 여기서 죽는다."""
    import json
    import torch

    meta = json.loads((model_dir / "meta.json").read_text())
    if meta.get("spec") != SPEC:
        raise RuntimeError(
            f"패치 필터 규격 불일치: 모델={meta.get('spec')} · 코드={SPEC}. "
            f"export_patch_clf.py 와 patch_filter.py 를 같이 갱신해야 한다.")
    if int(meta["patch"]) != PATCH or float(meta["spacing_mm"]) != SPACING:
        raise RuntimeError("패치 크기/spacing 이 코드 상수와 다르다.")
    net = build_net(int(meta["in_ch"]), int(meta.get("n_scalar", 0)))
    # weights_only=True: 순수 텐서 dict 만 읽는다. 지금 기본값(False)은 pickle 을 그대로
    # 실행하므로 안전하지 않고, torch 가 곧 기본을 뒤집을 예정이라 그때 동작이 바뀐다.
    # 명시해두면 지금도 안전하고 나중에도 안 깨진다(스모크에서 동작 확인).
    sd = torch.load(model_dir / "headA.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    return net, float(meta["threshold"])


def build_net(in_ch: int, n_scalar: int):
    """학습(`train_patch_classifier.build_model`)과 **완전히 같은 구조**여야 한다."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(
                    nn.Conv3d(i, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o), nn.LeakyReLU(0.01, True),
                    nn.Conv3d(o, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o), nn.LeakyReLU(0.01, True),
                    nn.MaxPool3d(2))
            self.enc = nn.Sequential(blk(in_ch, 16), blk(16, 32), blk(32, 64), blk(64, 128),
                                     nn.AdaptiveAvgPool3d(1), nn.Flatten())
            self.drop = nn.Dropout(0.5)
            self.headA = nn.Linear(128, 1)
            self.headB = nn.Sequential(nn.Linear(128 + n_scalar, 64),
                                       nn.LeakyReLU(0.01, True), nn.Linear(64, 1))

        def forward(self, x, scal=None):
            f = self.drop(self.enc(x))
            a = self.headA(f).squeeze(1)
            b = self.headB(torch.cat([f, scal], 1)).squeeze(1) if scal is not None else None
            return a, b
    return Net()


def _resample_patch(arr: np.ndarray, spacing_zyx, center_idx_zyx, is_label: bool):
    """extract_blob_patches._resample_patch 와 **같은 알고리즘**(값이 같아야 한다)."""
    from scipy import ndimage

    sp = np.asarray(spacing_zyx, float)
    half_mm = PATCH * SPACING / 2.0
    half_vox = np.ceil(half_mm / sp).astype(int) + 1
    lo = np.maximum(np.round(center_idx_zyx).astype(int) - half_vox, 0)
    hi = np.minimum(np.round(center_idx_zyx).astype(int) + half_vox + 1, arr.shape)
    sub = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if sub.size == 0:
        return np.zeros((PATCH,) * 3, dtype=np.float32)
    zoom = sp / SPACING
    out = ndimage.zoom(sub.astype(np.float32), zoom, order=0 if is_label else 1)
    c = (np.array(center_idx_zyx) - lo) * zoom
    res = np.zeros((PATCH,) * 3, dtype=np.float32)
    src_lo = np.round(c - PATCH / 2).astype(int)
    for a in range(3):
        src_lo[a] = int(np.clip(src_lo[a], 0, max(out.shape[a] - PATCH, 0)))
    src_hi = np.minimum(src_lo + PATCH, out.shape)
    piece = out[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    res[:piece.shape[0], :piece.shape[1], :piece.shape[2]] = piece
    return res


def blob_patches(seg_crop_zyx, img_norm_zyx, vessel_zyx, spacing_zyx, start_zyx):
    """blob 별 (4,64,64,64) 패치. 반환 (label_map, n, X)."""
    from scipy import ndimage
    from src import config, features

    lab, n = ndimage.label(seg_crop_zyx > 0)
    if n == 0:
        return lab, 0, np.zeros((0, 4, PATCH, PATCH, PATCH), np.float32)
    lut = features._TERR_LUT
    st = np.asarray(start_zyx, float)
    X = np.zeros((n, 4, PATCH, PATCH, PATCH), np.float32)
    for i in range(1, n + 1):
        ctr = np.argwhere(lab == i).mean(0) + st          # 원본 격자 좌표
        p_img = _resample_patch(img_norm_zyx, spacing_zyx, ctr, is_label=False)
        p_ves = _resample_patch(vessel_zyx, spacing_zyx, ctr, is_label=True)
        X[i - 1, 0] = p_img
        X[i - 1, 1] = (p_ves > 0).astype(np.float32)
        X[i - 1, 2] = (p_ves / float(N_VES)).astype(np.float32)
        X[i - 1, 3] = (lut[np.clip(p_ves.astype(int), 0, len(lut) - 1)]
                       / config.N_TERRITORY).astype(np.float32)
    return lab, n, X


def filter_blobs(seg_crop_zyx, net, threshold, img_norm_zyx, vessel_zyx,
                 spacing_zyx, start_zyx, device="cpu", bs=4):
    """임계 미만 blob 을 0 으로. 반환 (seg, n_blob, n_dropped, scores)."""
    import torch

    lab, n, X = blob_patches(seg_crop_zyx, img_norm_zyx, vessel_zyx, spacing_zyx, start_zyx)
    if n == 0:
        return seg_crop_zyx, 0, 0, np.zeros(0)
    net = net.to(device)
    sc = []
    with torch.no_grad():
        for i in range(0, n, bs):
            a, _ = net(torch.from_numpy(X[i:i + bs]).to(device))
            sc.append(torch.sigmoid(a).cpu().numpy())
    sc = np.concatenate(sc)
    out = seg_crop_zyx.copy()
    drop = 0
    for i in range(1, n + 1):
        if sc[i - 1] < threshold:
            out[lab == i] = 0
            drop += 1
    return out, n, drop, sc
