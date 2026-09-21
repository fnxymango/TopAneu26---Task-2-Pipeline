"""
TopAneu 모달리티 적응형 정규화 (우리 EDA 기반, per-image).
전역 fingerprint 통계(intensityproperties)를 쓰지 않는다 — CTA+MRA 혼합에서
그 값은 무의미한 blend이기 때문. per-image로 모달리티 감지 후 개별 정규화.

설계 근거 (analysis/topaneu_eda_per_case.csv):
  - CTA 전체 mean이 공기/padding(-1000~-2048)에 지배돼 케이스간 CV 0.84로 불안정
    → CT도 '전경(>-100) z-score' 로 안정화 (jskim은 CT 전역 z-score 사용, 우리는 전경).
  - CTA 동맥류/혈관 ~80~440 HU, 뼈 600+ → clip [-200, 800] (혈관 대역 보존 + 뼈 억제).
  - MRA 혈관은 극단 밝은 꼬리(동맥류 p95~2004) → 상한 99.9로 완화해 혈관 보존
    (jskim 99.5는 혈관 꼬리를 자를 수 있음).
  - 모달리티 감지: MRA min=0(magnitude), CTA는 강한 음수(공기) → (<-300) 비율로 판별.
"""
import numpy as np
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization


class TopAneuAdaptiveNorm(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    CT_CLIP = (-200.0, 800.0)   # 혈관 대역 보존, 극단 뼈/공기 억제
    CT_FG = -100.0              # 전경(뇌/조직/혈관) 기준 — 공기 제외
    MR_PCT = (0.5, 99.9)        # 혈관 밝은 꼬리 보존

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        img = image.astype(self.target_dtype, copy=False)
        eps = 1e-8

        # per-image 모달리티 감지 (전역 통계 미사용)
        is_ct = float((img < -300).mean()) > 0.01

        if is_ct:  # ── CTA ──
            np.clip(img, self.CT_CLIP[0], self.CT_CLIP[1], out=img)
            fg = img[img > self.CT_FG]
            m = float(fg.mean()) if fg.size else float(img.mean())
            s = float(fg.std()) if fg.size else float(img.std())
            img -= m
            img /= max(s, eps)
            return img

        # ── MRA ── (배경 0 유지, 전경만 정규화)
        fgmask = img > 0
        out = np.zeros(img.shape, dtype=self.target_dtype)
        if fgmask.sum() > 0:
            vals = img[fgmask]
            lo, hi = np.percentile(vals, self.MR_PCT)
            v = np.clip(vals, lo, hi)
            m = float(v.mean())
            s = float(v.std())
            out[fgmask] = (v - m) / max(s, eps)
        return out
