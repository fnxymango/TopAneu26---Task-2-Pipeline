"""파생 입력 채널 생성기 (아이디어 1·2·3 채널).

핵심 원칙: **학습 전처리와 추론에서 완전히 동일한 함수**를 쓴다.
  - 학습:  vessel = GT vessel_mask
  - 추론:  vessel = Stage-1 vessel 모델(Dataset011)의 예측 vessel_mask
따라서 채널 생성 로직은 vessel_mask 의 출처와 무관하게 여기 한 곳에만 존재한다.

생성 채널 (config.ACTIVE_DERIVED_CHANNELS 로 선택):
  vessel_proximity : exp(-d/tau).  d = 각 voxel 에서 최근접 vessel voxel 까지의 거리(mm).
                     -> vessel 근처일수록 1. 동맥류는 vessel 위에 생기므로 강한 위치 prior.
  territory        : 최근접 vessel segment 의 해부학 대분류(VB/PCA/ICA/ACA/MCA) /5.
                     -> location 50-class 의 대분류와 정렬된 '대략적 해부학 위치'.
  laterality       : 최근접 vessel segment 의 좌/우 부호 {-1(L), 0(NA), +1(R)}.
                     -> 50-class 가 좌우 쌍으로 구성되어 있으므로 직접적 힌트.
  modality         : CT=1 / MR=0 상수맵 (아이디어 3의 단순 채널판).
  symmetry         : 정중시상면 기준 반전 후 강도 비대칭(부호 있음). 동맥류의 국소적
                     비대칭(굵기/형태 차이)을 강조 (아이디어 2).

기하 정합: 모든 채널은 image 와 동일 grid 에서 계산되고 image 와 동일 affine/header 로 저장.
vessel_mask 는 image 와 shape/spacing 이 동일함을 이미 확인함.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, center_of_mass, shift as ndi_shift

from . import config


# ---------------------------------------------------------------------------
# NIfTI I/O 헬퍼
# ---------------------------------------------------------------------------
def load_nifti(path):
    """(array[float32], nib_image) 반환. array 는 파일에 저장된 축 순서 그대로."""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    return arr, img


def spacing_of(img) -> tuple[float, float, float]:
    return tuple(float(z) for z in img.header.get_zooms()[:3])


def lr_axis_of(img) -> int:
    """좌우(L-R, world x축)에 가장 가까운 array 축 인덱스. affine 으로 판별(LPS/RAS 무관)."""
    aff = img.affine[:3, :3]
    return int(np.argmax(np.abs(aff[0])))


def save_like(array: np.ndarray, ref_img, out_path):
    """array 를 ref_img 와 동일한 affine/header(float32)로 저장."""
    out = nib.Nifti1Image(array.astype(np.float32, copy=False), ref_img.affine, ref_img.header)
    out.set_data_dtype(np.float32)
    nib.save(out, str(out_path))


# ---------------------------------------------------------------------------
# vessel 라벨 -> territory / laterality LUT (벡터화용)
# ---------------------------------------------------------------------------
def _territory_luts():
    terr = np.zeros(37, dtype=np.float32)   # index 0..36
    lat = np.zeros(37, dtype=np.float32)
    for label, (t, l) in config.VESSEL_TERRITORY.items():
        terr[label] = t
        lat[label] = l
    return terr, lat


_TERR_LUT, _LAT_LUT = _territory_luts()


# ---------------------------------------------------------------------------
# 개별 feature 계산
# ---------------------------------------------------------------------------
def _vessel_distance_and_nearest(vessel: np.ndarray, spacing):
    """EDT 한 번으로 (최근접 vessel 까지 거리[mm], 최근접 vessel 라벨) 동시 계산."""
    binary = vessel > 0
    if not binary.any():
        # vessel 이 아예 없는 극단 케이스 방어
        dist = np.full(vessel.shape, 1e4, dtype=np.float32)
        nearest = np.zeros(vessel.shape, dtype=vessel.dtype)
        return dist, nearest
    bg = ~binary
    dist, inds = distance_transform_edt(bg, sampling=spacing, return_indices=True)
    nearest = vessel[tuple(inds)]
    return dist.astype(np.float32), nearest


def feat_vessel_proximity(dist_mm: np.ndarray, tau_mm: float = 3.0) -> np.ndarray:
    return np.exp(-dist_mm / tau_mm).astype(np.float32)


def brain_mask_from_edt(dist_mm: np.ndarray, nearest_label: np.ndarray,
                        radius_mm: float | None = None,
                        seed_exclude: set[int] | None = None) -> np.ndarray:
    """'최근접 vessel 까지 거리 <= R' 로 뇌 실질을 근사한 대략적 뇌 마스크.

    **추가 계산이 없다.** `_vessel_distance_and_nearest` 가 이미 만들어 캐시해 둔
    (거리, 최근접라벨) 을 그대로 재사용한다 — EDT 를 다시 돌지 않는다.

    `seed_exclude` 는 '그 라벨이 최근접인 voxel 을 뺀다' 로 구현한다. 엄밀히는
    '제외 라벨을 뺀 혈관까지의 거리 <= R' 과 다르지만(그건 EDT 를 한 번 더 돌아야
    한다 — 최대 케이스에서 수십 초, 지금 7분 한도와 정면으로 부딪친다), 우리가 막으려는
    누출에는 정확히 같게 동작한다: 목 voxel 의 최근접은 ICA-C1-C5, 안와 voxel 의
    최근접은 OA 이므로 그대로 걸러진다.

    형태학 후처리(구멍 메우기/최대 연결성분)는 **일부러 안 한다**. 목적이 분할이
    아니라 mean/std 표본이라 구멍이 있어도 통계가 흔들리지 않고, 최대 케이스
    1.3억 voxel 에서 `binary_fill_holes` 는 런타임 예산을 잡아먹는다.
    """
    if radius_mm is None:
        radius_mm = config.BRAIN_MASK_RADIUS_MM
    if seed_exclude is None:
        seed_exclude = config.BRAIN_MASK_SEED_EXCLUDE

    mask = dist_mm <= radius_mm
    if seed_exclude:
        mask &= ~np.isin(nearest_label, list(seed_exclude))
    return mask


def _coarsen(image: np.ndarray, spacing, target_mm: float = 2.0):
    """통계용 저해상도 격자로 솎아낸다. (coarse_image, coarse_spacing, factors) 반환.

    mean/std 와 형태학은 해상도가 필요 없다. 최대 케이스 1.3억 voxel 에서 원 해상도로
    `binary_fill_holes`/`label` 을 돌리면 GC 7분 예산을 통째로 날린다. 2mm 격자면
    voxel 이 수십 배 줄어 형태학이 사실상 공짜가 되고, 표본 통계는 변하지 않는다.
    """
    f = [max(1, int(round(target_mm / s))) for s in spacing]
    return image[::f[0], ::f[1], ::f[2]], [s * k for s, k in zip(spacing, f)], f


def brain_mask_intensity(image: np.ndarray, spacing, is_ct: bool,
                         erode_mm: float = 8.0, target_mm: float = 2.0):
    """강도 기반 **대략적 뇌 실질 마스크**. vessel 예측에 의존하지 않는다.

    `brain_mask_from_edt` 의 약점이 vessel 예측 실패를 그대로 물려받는 것이라
    (실측 점유율 0.126~0.672 로 5배 흔들림), 영상 강도만으로 독립적으로 잡는 판이다.

    modality 별로 규칙이 다르다 — 같은 임계를 쓸 수 없다:
      CT : HU 라서 절대 임계가 통한다. 뇌 실질은 대략 [-20, 100] HU.
           **조영된 혈관(100~400 HU)은 일부러 뺀다** — 평가 기준 조직이 혈관이라
           마스크에 혈관을 넣으면 순환논증이 된다. 두개골(>300)·공기(<-200)도 빠진다.
      MR : 임의 단위라 절대 임계가 없다. Otsu 로 공기/조직을 가른다.

    두피·두개골 테두리는 `erode_mm` 만큼 침식해 떼어낸다.

    반환: (coarse_mask, coarse_image) — 둘 다 저해상도 격자. 통계만 쓸 것이므로
    원 해상도로 되돌리지 않는다(되돌리는 zoom 이 형태학보다 비싸다).
    """
    from scipy import ndimage

    img, sp, _ = _coarsen(image, spacing, target_mm)

    if is_ct:
        core = (img > -20) & (img < 100)
    else:
        from skimage.filters import threshold_otsu
        nz = img[img > 0]
        if nz.size == 0:
            return np.zeros(img.shape, dtype=bool), img
        try:
            thr = float(threshold_otsu(nz))
        except ValueError:
            thr = float(np.percentile(nz, 50))
        core = img > thr

    if not core.any():
        return core, img

    core = ndimage.binary_fill_holes(core)
    lab, n = ndimage.label(core)
    if n > 1:                                  # 두피 조각·테이블·잡음 성분을 떨군다
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        core = lab == int(np.argmax(sizes))

    it = max(1, int(round(erode_mm / max(min(sp), 1e-6))))
    eroded = ndimage.binary_erosion(core, iterations=it)
    # 침식이 다 갉아먹으면(얇은 FOV) 침식 전으로 되돌린다 — 빈 마스크보다 낫다.
    return (eroded if eroded.any() else core), img


def zscore_in_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 **안쪽 통계**로 볼륨 전체를 z-score. 바깥도 같은 (mean,std) 로 변환한다.

    바깥을 0 으로 지우지 않는 이유: 두개골/안면 영역이 완전히 무의미하지는 않고,
    무엇보다 nnU-Net 의 use_mask_for_norm 경로(바깥을 0 으로 만드는)와 달리 여기서는
    **어떤 정보도 버리지 않는다**. 바뀌는 건 '어느 voxel 로 mean/std 를 재는가' 하나뿐이라
    센터 간 이질성만 걷어내고 나머지는 기존과 동일하다.

    마스크가 비면(vessel 예측 실패 등) 전체 볼륨 통계로 조용히 되돌아간다 —
    정규화가 통째로 깨지는 것보다 기존 동작으로 떨어지는 편이 낫다.
    """
    img = image.astype(np.float32, copy=True)
    sample = img[mask] if mask.any() else img
    mean = float(sample.mean())
    std = float(sample.std())
    img -= mean
    img /= max(std, 1e-8)
    return img


def feat_territory(nearest_label: np.ndarray) -> np.ndarray:
    return (_TERR_LUT[nearest_label] / config.N_TERRITORY).astype(np.float32)


def feat_laterality(nearest_label: np.ndarray) -> np.ndarray:
    return _LAT_LUT[nearest_label].astype(np.float32)


def feat_modality(shape, is_ct: bool) -> np.ndarray:
    return np.full(shape, 1.0 if is_ct else 0.0, dtype=np.float32)


def feat_symmetry(image: np.ndarray, vessel: np.ndarray, lr_axis: int,
                  clip_percentiles=(0.5, 99.5)) -> np.ndarray:
    """정중시상면 기준 반전 비대칭 (부호 있음).

    - 강도를 percentile 로 [0,1] 정규화 (CT/MR 스케일 차이 흡수).
    - 정중선은 vessel mask 의 L-R 축 무게중심으로 추정(혈관 트리는 대략 좌우 대칭).
      vessel 이 없으면 기하 중심 사용.
    - 반전 영상을 정중선 기준으로 정렬 후 (원본 - 반전) 을 반환.
    주의: 회전 보정 없는 근사(rigid flip). 큰 두부 기울기에는 v2 에서 정합 추가 예정.
    """
    lo, hi = np.percentile(image, clip_percentiles)
    imn = np.clip((image - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)

    n = image.shape[lr_axis]
    if (vessel > 0).any():
        mid = float(center_of_mass(vessel > 0)[lr_axis])
    else:
        mid = (n - 1) / 2.0

    flipped = np.flip(imn, axis=lr_axis)
    shift_vec = [0.0, 0.0, 0.0]
    shift_vec[lr_axis] = 2.0 * mid - (n - 1)         # 정중선 정렬 shift
    flipped = ndi_shift(flipped, shift_vec, order=1, mode="nearest")

    return (imn - flipped).astype(np.float32)


# ---------------------------------------------------------------------------
# 통합 진입점
# ---------------------------------------------------------------------------
def build_channels(image: np.ndarray, vessel: np.ndarray, img_ref,
                   is_ct: bool, feature_names: list[str] | None = None,
                   tau_mm: float = 3.0) -> dict[str, np.ndarray]:
    """요청한 파생 채널들을 이름->array(float32) dict 로 반환.

    image, vessel: 동일 grid 의 numpy 배열.
    img_ref: nibabel image (spacing/affine 참조용).

    `config.BRAIN_NORM` 이 켜져 있으면 반환 dict 에 **`"image"` 키가 추가**된다 —
    뇌 마스크 안쪽 통계로 z-score 한 ch0 이다. 호출부는 이 키가 있으면 원본 image
    대신 그것을 저장/입력해야 한다(없으면 기존대로 원본을 쓴다).
    """
    if feature_names is None:
        feature_names = [name for name, _ in config.ACTIVE_DERIVED_CHANNELS]

    spacing = spacing_of(img_ref)
    lr_axis = lr_axis_of(img_ref)

    # EDT 결과는 여러 채널이 공유 -> 필요할 때 한 번만 계산
    _cache: dict[str, tuple] = {}

    def edt():
        if "edt" not in _cache:
            _cache["edt"] = _vessel_distance_and_nearest(vessel, spacing)
        return _cache["edt"]

    out: dict[str, np.ndarray] = {}
    for name in feature_names:
        if name == "vessel_proximity":
            dist, _ = edt()
            out[name] = feat_vessel_proximity(dist, tau_mm)
        elif name == "territory":
            _, nearest = edt()
            out[name] = feat_territory(nearest)
        elif name == "laterality":
            _, nearest = edt()
            out[name] = feat_laterality(nearest)
        elif name == "modality":
            out[name] = feat_modality(image.shape, is_ct)
        elif name == "symmetry":
            out[name] = feat_symmetry(image, vessel, lr_axis)
        else:
            raise ValueError(f"알 수 없는 feature: {name}")

    # ch0 뇌 마스크 정규화가 켜져 있으면 정규화된 image 도 같이 돌려준다.
    # nnU-Net 이 아니라 여기서 하는 이유는 config.BRAIN_NORM 주석 참조.
    # 위 EDT 캐시를 그대로 재사용하므로 EDT 를 다시 돌지 않는다.
    if config.BRAIN_NORM:
        dist, nearest = edt()
        out["image"] = zscore_in_mask(image, brain_mask_from_edt(dist, nearest))
    return out


def build_channels_from_paths(image_path, vessel_path, is_ct: bool,
                              feature_names: list[str] | None = None):
    """파일 경로로부터 채널 생성. (image_array, img_ref, {name: array}) 반환."""
    image, img_ref = load_nifti(image_path)
    vessel, _ = load_nifti(vessel_path)
    vessel = vessel.astype(np.int16, copy=False)
    if vessel.shape != image.shape:
        raise ValueError(f"geometry 불일치: image{image.shape} vs vessel{vessel.shape}")
    chans = build_channels(image, vessel, img_ref, is_ct, feature_names)
    return image, img_ref, chans
