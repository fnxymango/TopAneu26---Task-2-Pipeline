"""TopAneu 파이프라인 전역 설정.

- 경로: 원본 데이터셋(읽기 전용)과 로컬 작업 폴더(쓰기)를 명확히 분리.
- 데이터셋 ID 규약, modality 판별, canonical case 순서/ID 매핑.
- vessel 라벨 -> (해부학 territory, laterality) 테이블 (아이디어 1의 위치 임베딩 근거).
- split 로딩 유틸.

주의: 여기서 정의하는 canonical case 순서(sorted images/*.nii.gz)는 기존
convert_to_nnunet.py 와 동일한 규약이라 Dataset001 의 TopAneu_XXX ID 와 일치한다.
따라서 모든 커스텀 데이터셋(002/011 ...)이 같은 케이스에 같은 nnU-Net ID를 부여한다.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
# 이 파일 기준으로 작업 루트(jslee/)를 자동 계산 -> 폴더를 옮겨도 동작.
ROOT = Path(__file__).resolve().parent.parent          # /home/user/TopAneu/seg/jslee
DATA_ROOT = Path("/home/user/dataset/TopAneu")          # 원본 = **batch2** (읽기 전용!)

# batch1(2026-06-15 릴리스) 아카이브. 2026-07-31 batch2 가 나오면서 위 DATA_ROOT 의 내용이
# 통째로 교체되고 batch1 은 여기로 밀려났다. 이 경로가 필요한 이유:
#   Dataset001/002/003/011 의 imagesTr·labelsTr 은 DATA_ROOT 를 향한 심볼릭 링크라
#   교체 순간 (a) 개명된 center4 11케이스가 dangling 이 되고 (b) 나머지는 batch2 GT 를
#   가리키게 됐다. 즉 batch1 로 학습한 런을 batch2 GT 로 채점하는 상태가 된다.
#   repair_batch1_links.py 가 이 상수로 링크를 되돌린다.
DATA_ROOT_B1 = Path("/home/user/dataset/TopAneu_old")

IMAGES_DIR = DATA_ROOT / "images"
VESSEL_MASKS_DIR = DATA_ROOT / "vessel_masks"
TYPE_MASKS_DIR = DATA_ROOT / "type_masks"
LOCATION_MASKS_DIR = DATA_ROOT / "location_masks"

SPLIT_JSON = DATA_ROOT / "dataset_split.json"
LOCATION_MAPPING_JSON = DATA_ROOT / "location_mapping.json"
VESSEL_MAPPING_JSON = DATA_ROOT / "vessel_mapping.json"
TYPE_MAPPING_JSON = DATA_ROOT / "type_mapping.json"

# nnU-Net 로컬 디렉토리 (env.sh 에서 export 하는 값과 반드시 동일해야 함)
NNUNET_RAW = ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = ROOT / "nnUNet_results"

IMAGE_SUFFIX = "_0000.nii.gz"
MASK_SUFFIX = ".nii.gz"

# ---------------------------------------------------------------------------
# 데이터셋 ID 규약
# ---------------------------------------------------------------------------
#: batch1(98케이스 · location 50-class) 로 만들어진 기존 데이터셋. **더 이상 새로 만들지 않는다.**
DATASET_LOCATION_BASELINE = 1   # Dataset001_TopAneu           (image -> 50-class location)
DATASET_LOCATION_MC = 2         # Dataset002_TopAneuMC         (Stage A: 멀티채널 -> 50-class location)
DATASET_LOCATION_ROI = 3        # Dataset003_TopAneuROI        (Stage B: vessel-guided ROI 크롭 멀티채널)
DATASET_VESSEL = 11             # Dataset011_TopAneuVessel     (Stage-1: image -> 36-class vessel)

#: batch2(417케이스 · location **52**-class · 음성 111케이스 포함) 재구축본. 2xx = batch 2.
#  batch1 데이터셋을 덮어쓰지 않고 새 ID 로 만드는 이유:
#    1) exp_01~14 의 결과/체크포인트를 그대로 보존해야 한다(사용자 지시).
#    2) 라벨 의미가 바뀌었다. batch2 가 47/48 에 'M1 early bifurcation' 을 새로 끼워 넣어
#       기존 47~50(M1-M2 junction, Distal-M2M3)이 49~52 로 밀렸다. 같은 폴더에 섞으면
#       클래스 인덱스가 조용히 어긋난다.
#    3) 케이스 수가 98 -> 417 이라 TopAneu_XXX 일련번호 자체가 전부 재배정된다.
DATASET_B2_LOCATION_BASELINE = 201   # Dataset201_TopAneuB2
DATASET_B2_LOCATION_MC = 202         # Dataset202_TopAneuMCB2
DATASET_B2_LOCATION_ROI = 203        # Dataset203_TopAneuROIB2
DATASET_B2_VESSEL = 211              # Dataset211_TopAneuVesselB2

#: **예측 vessel** 로 파생채널을 다시 만든 재빌드본 (2026-08-04~).
#  D202/D203 은 파생채널을 **GT vessel mask** 로 만들었다. GT vessel 은 추론 때 존재하지
#  않으므로(주최측이 image 만 준다) 그 두 데이터셋은 **상한 베이스라인**이지 제출 경로가
#  아니다. 학습은 완벽한 혈관을, 추론은 D211 이 예측한 혈관을 보게 되는 분포차가 그대로
#  성능 손실로 나타난다. 여기서는 학습 채널도 D211 예측으로 만들어 그 차이를 없앤다.
#  D202 <-> D212, D203 <-> D213 은 **vessel 출처만 다르고 나머지는 동일**해야 한다 —
#  그래야 둘의 차이가 곧 'GT vessel 이 주던 이득'의 크기가 된다.
DATASET_B2_LOCATION_MC_PV = 212      # Dataset212_TopAneuMCB2PV
DATASET_B2_LOCATION_ROI_PV = 213     # Dataset213_TopAneuROIB2PV

#: D213 과 **ch0 정규화만** 다른 판 (2026-08-07). BRAIN_NORM=1 로 빌드해야 한다.
#  ch0 을 볼륨 전체 통계가 아니라 vessel EDT 뇌 마스크 안쪽 통계로 z-score 한다
#  (근거·실측: EXPERIMENTS_NORM.md — 피험자 간 산포 CT -31% / MR -9%, 센터 간 MR -73%).
#  크롭 bbox 는 같은 예측 vessel + 같은 MIN_CC 로 잡히므로 D213 과 **기하가 동일**하다.
#  따라서 D213 vs D214 의 차이는 정규화 단독 효과다. 반드시 같은 trainer/epoch 로 읽을 것.
#  주의: 이 데이터셋으로 학습한 모델은 **추론 때도 BRAIN_NORM=1** 이어야 한다. 학습·추론이
#  같은 features.build_channels 를 쓰므로 환경변수만 맞추면 되지만, 끄고 추론하면 조용히
#  분포가 어긋난다(dataset.json 의 ch0 이 'nonorm' 이라 nnU-Net 이 보정해주지 않는다).
DATASET_B2_LOCATION_ROI_PV_BN = 214  # Dataset214_TopAneuROIB2PVBN

#: D213 과 **ch0 정규화만** 다른 두 번째 판 (2026-08-09). ch0 = (x-median)/(IQR/1.349).
#  왜 D214(뇌마스크 z-score)로 부족한가 — 실측하니 D214 도 CT 를 못 살린다: 동맥류 값이
#  CT 0.41 / MR 6.23(비율 0.07)로 D213(0.10)보다 오히려 나쁘다. 마스크로 통계 영역을
#  좁혀도 그 안의 뼈/석회화가 std 를 다시 지배하기 때문. IQR 은 정의상 그 꼬리에 둔감해
#  CT 5.94 / MR 5.35(비율 1.11)로 정렬된다. 근거·전체 표: src/robust_norm.py 헤더.
#  빌드는 src/build_d215_robustnorm.py — ch0 만 새로 쓰고 ch1~5/labels 는 D213 심볼릭
#  링크라 기하가 바이트 단위로 동일하다(= 정규화 단독 효과).
#  주의: ch0 이 'nonorm' 이므로 **추론도 robust_norm.robust_normalize 를 통과**해야 한다.
DATASET_B2_LOCATION_ROI_PV_RN = 215  # Dataset215_TopAneuROIB2PVRN

#: D214 + **정규화 좌표 3채널** (2026-08-11). ch0~5/labels 는 D214 심볼릭 링크라 기하가
#  바이트 단위로 동일하고, 늘어난 건 좌표 램프 3개(케이스당 3.8MB)뿐이다.
#  왜 — patch_size [96,128,160] 이 크롭 볼륨의 약 3% 라 네트워크가 자기 위치를 모른다.
#  위치 정보는 territory(5분류)와 laterality 뿐인데 laterality 는 동맥류 위치 정확도 43.2%.
#  반면 밖에서 계산한 좌표만으로 52-way top1 38.6%·측 93.3%·territory 86.4% 가 나온다.
#  blob 단위 사후 결합의 천장(좌표+U-Net 합집합 52.3%)을 넘을 유일한 후보다.
#  근거·측정: EXPERIMENTS_MCC.md, 빌드: src/build_d216_coords.py
DATASET_B2_LOCATION_ROI_PV_BN_XYZ = 216  # Dataset216_TopAneuROIB2PVBNXYZ

DATASET_B2_ROI_PV2_BN = 217
DATASET_B2_ROI_PV2_BN_XYZ = 218
DATASET_B2_VESSEL_BN = 219

DATASET_NAMES = {
    DATASET_LOCATION_BASELINE: "Dataset001_TopAneu",
    DATASET_LOCATION_MC: "Dataset002_TopAneuMC",
    DATASET_LOCATION_ROI: "Dataset003_TopAneuROI",
    DATASET_VESSEL: "Dataset011_TopAneuVessel",
    DATASET_B2_LOCATION_BASELINE: "Dataset201_TopAneuB2",
    DATASET_B2_LOCATION_MC: "Dataset202_TopAneuMCB2",
    DATASET_B2_LOCATION_ROI: "Dataset203_TopAneuROIB2",
    DATASET_B2_VESSEL: "Dataset211_TopAneuVesselB2",
    DATASET_B2_LOCATION_MC_PV: "Dataset212_TopAneuMCB2PV",
    DATASET_B2_LOCATION_ROI_PV: "Dataset213_TopAneuROIB2PV",
    DATASET_B2_LOCATION_ROI_PV_BN: "Dataset214_TopAneuROIB2PVBN",
    DATASET_B2_LOCATION_ROI_PV_RN: "Dataset215_TopAneuROIB2PVRN",
    DATASET_B2_LOCATION_ROI_PV_BN_XYZ: "Dataset216_TopAneuROIB2PVBNXYZ",
    # 2세대 예측 vessel(PV2 = exp_44/D211@1250ep farm, test dice 0.615 -> 0.745)로 다시 빌드한
    # D214/D216 대응본. 레시피는 한 글자도 안 바꾸고 **vessel 출처만** 다르다 — 그래야
    # exp_38/39/40 과의 차이가 곧 'vessel 품질을 올린 효과'가 된다.
    DATASET_B2_ROI_PV2_BN: "Dataset217_TopAneuROIB2PV2BN",
    DATASET_B2_ROI_PV2_BN_XYZ: "Dataset218_TopAneuROIB2PV2BNXYZ",
    # D211(vessel)에서 **ch0 정규화만** 바꾼 판 — modality 별 뇌마스크 기준 z-score.
    # CT 가 미학습 케이스에서 무너지는 문제(test dice MR 0.834 vs CT 0.499)의 대응.
    DATASET_B2_VESSEL_BN: "Dataset219_TopAneuVesselB2BN",
}

#: 예측 vessel 재빌드본 -> 대응하는 GT vessel 원본. 표에서 짝지어 읽기 위한 것이다.
PV_TO_GT = {
    DATASET_B2_LOCATION_MC_PV: DATASET_B2_LOCATION_MC,
    DATASET_B2_LOCATION_ROI_PV: DATASET_B2_LOCATION_ROI,
}

#: D211 이 예측한 vessel mask 를 **원본 case id** 이름으로 모아 두는 폴더.
#  nnU-Net 추론 산출물은 nnU-Net id(TopAneu_000.nii.gz)로 나오는데 build_*_dataset 의
#  --vessel-dir 는 원본 case id(topaneu_center1_mr_001.nii.gz)를 찾는다. 이름이 다르면
#  조용히 FileNotFoundError 가 아니라 **엉뚱한 케이스의 혈관**을 붙일 위험이 있으므로
#  중간에 심볼릭 링크 farm 을 한 번 거친다(run_predict_vessel_b2.sh 가 만든다).
#  **DATA_ROOT 밑에 두지 않는다** — 그쪽은 주최측 원본이라 쓰기 금지다(쓸 수는 있어도
#  원본과 우리 산출물이 섞이면 다음 batch 릴리스 때 무엇이 원본인지 구분할 수 없다).
VESSEL_MASKS_PRED_B2_DIR = ROOT / "predictions" / "vessel_b2_by_case"

#: batch1 데이터셋 -> 대응하는 batch2 데이터셋 (표/스크립트에서 세대 짝을 찾을 때 사용).
B1_TO_B2 = {
    DATASET_LOCATION_BASELINE: DATASET_B2_LOCATION_BASELINE,
    DATASET_LOCATION_MC: DATASET_B2_LOCATION_MC,
    DATASET_LOCATION_ROI: DATASET_B2_LOCATION_ROI,
    DATASET_VESSEL: DATASET_B2_VESSEL,
}
B2_DATASETS = set(B1_TO_B2.values())


def dataset_dir(dataset_id: int, base: Path = NNUNET_RAW) -> Path:
    return base / DATASET_NAMES[dataset_id]


# ---------------------------------------------------------------------------
# 라벨링 오류 확인 케이스 (주최측 확인, batch2 에서 수정 예정) - 제외하지 않고 표시만
# ---------------------------------------------------------------------------
KNOWN_LATERALITY_ISSUE_CASES = {
    "topaneu_center2_mr_032",
    "topaneu_center2_mr_085",
    "topaneu_center4_ct_005",
    "topaneu_center4_ct_102",
}


# ---------------------------------------------------------------------------
# canonical case 순서 / nnU-Net ID 매핑
# ---------------------------------------------------------------------------
def get_case_id_from_image(image_filename: str) -> str:
    base = image_filename.replace(".nii.gz", "")
    if base.endswith("_0000"):
        base = base[: -len("_0000")]
    return base


def list_case_ids() -> list[str]:
    """images 폴더를 정렬해 원본 case_id 리스트 반환 (convert_to_nnunet.py 와 동일 규약)."""
    files = sorted(glob.glob(str(IMAGES_DIR / "*.nii.gz")))
    return [get_case_id_from_image(os.path.basename(f)) for f in files]


def nnunet_id(index: int) -> str:
    return f"TopAneu_{index:03d}"


#: 실제로 데이터셋을 만들 때 쓴 nid 배정 기록. **이게 정본이다.**
CASE_ID_MAPPING_JSON = ROOT / "nnUNet_raw/Dataset214_TopAneuROIB2PVBN/case_id_mapping.json"


def case_to_nnunet_map() -> dict[str, str]:
    """원본 case_id -> nnU-Net ID (TopAneu_XXX). 모든 데이터셋이 공유하는 canonical 매핑.

    ★ 순서 기반 재계산은 **케이스가 하나라도 빠지면 그 뒤가 전부 밀린다.**
      2026-08-14 에 topaneu_center1_mr_150 이 삭제되면서 416개를 다시 세면 386개가
      한 칸씩 밀린다. 2026-08-22 에 이걸로 D214 crop_meta 83개가 잘못된 nid 로
      덮어써졌다(D213 에서 복구, 손상본은 crop_meta.damaged_20260822/ 에 보존).
      같은 사고가 그 전에도 실험 3개를 무효화한 적이 있다.

    그래서 **빌드 당시 기록(case_id_mapping.json)을 먼저 읽는다.** 그 파일이 정본이고,
    순서 재계산은 기록이 아직 없는 최초 빌드에서만 쓴다.
    """
    if CASE_ID_MAPPING_JSON.exists():
        with open(CASE_ID_MAPPING_JSON) as f:
            rec = json.load(f)
        # ★ 기록에는 있지만 **지금 디스크에 없는** 케이스는 뺀다.
        #   2026-08-14 에 topaneu_center1_mr_150 이 삭제됐다. 기록 그대로 돌려주면
        #   417개가 되어 D211(416개) 같은 기존 데이터셋과 집합이 어긋나고,
        #   split 을 순회하는 코드는 없는 파일을 읽으려다 죽는다(실제로 D219 빌드와
        #   mirroring 대조가 여기서 멈췄다). nid 배정 자체는 기록을 따르되 존재하는
        #   케이스만 남기는 게 두 요구를 동시에 만족한다.
        live = set(list_case_ids())
        m = {d["original_case_id"]: d["new_id"] for d in rec
             if d["original_case_id"] in live}
        # 기록에 없는 케이스(새로 들어온 것)만 순서 규칙으로 보충한다 — 기존 배정은 안 건드린다.
        used = set(m.values())
        nxt = 0
        for cid in list_case_ids():
            if cid in m:
                continue
            while nnunet_id(nxt) in used:
                nxt += 1
            m[cid] = nnunet_id(nxt); used.add(m[cid])
        return m
    return {cid: nnunet_id(i) for i, cid in enumerate(list_case_ids())}


def modality_of(case_id: str) -> str:
    """'ct' 또는 'mr'. 파일명 규약 topaneu_{center}_{modality}_{id} 사용."""
    parts = case_id.split("_")
    return parts[2] if len(parts) > 2 else "unknown"


def is_ct(case_id: str) -> bool:
    return modality_of(case_id) == "ct"


# ---------------------------------------------------------------------------
# split 로딩
# ---------------------------------------------------------------------------
def load_split(split_json: Path = None) -> dict[str, list[str]]:
    """dataset_split.json -> {'train': [...], 'val': [...], 'test': [...]} (원본 case_id).

    batch1 은 키가 `split_ids`, batch2 는 `splits` 다. 둘 다 받는다 — 형식만 다르고
    의미는 같은데 여기서 KeyError 로 죽으면 배치 전환 때마다 호출부를 다 고쳐야 한다.
    """
    with open(split_json or SPLIT_JSON) as f:
        d = json.load(f)
    key = "splits" if "splits" in d else "split_ids"
    return {k: list(d[key][k]) for k in ("train", "val", "test")}


def load_case_meta(split_json: Path = None) -> dict[str, dict]:
    """case_id -> 케이스 메타(center/modality/locations/n_aneurysms/split). batch2 전용.

    batch2 dataset_split.json 의 `cases` 블록에만 있는 정보다(batch1 에는 없다).
    음성 케이스(n_aneurysms == 0) 선별에 쓴다.
    """
    with open(split_json or SPLIT_JSON) as f:
        d = json.load(f)
    if "cases" not in d:
        return {}
    out = {}
    for sp in ("train", "val", "test"):
        for c in d["cases"][sp]:
            out[c["case_id"]] = {**c, "split": sp}
    return out


def negative_case_ids(split_json: Path = None) -> dict[str, list[str]]:
    """동맥류가 하나도 없는 케이스를 split 별로. batch2 에서 처음 생겼다(111개, 전부 center1 MRA).

    FP 억제 측정의 근거 집합 — 이 케이스들에서 나오는 모든 전경 예측은 정의상 false positive 다.
    """
    meta = load_case_meta(split_json)
    out = {"train": [], "val": [], "test": []}
    for cid, m in meta.items():
        if m.get("n_aneurysms", 0) == 0:
            out[m["split"]].append(cid)
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 라벨 매핑 로딩
# ---------------------------------------------------------------------------
def load_labels(mapping_json: Path) -> dict[str, int]:
    with open(mapping_json) as f:
        d = json.load(f)
    return d["labels"] if "labels" in d else d


# ---------------------------------------------------------------------------
# vessel 라벨(36-class) -> 해부학 territory / laterality
# ---------------------------------------------------------------------------
# territory: 1=VB(vertebro-basilar), 2=PCA, 3=ICA, 4=ACA, 5=MCA
#   -> location 50-class 의 대분류(1_VB..5_MCA) 및 split 의 region_by_case 와 정렬됨.
# laterality: +1=Right, -1=Left, 0=NA/midline
# (아이디어 1: vessel 은 해부학 순서를 따르므로, 가장 가까운 vessel segment 의
#  territory/laterality 를 voxel 마다 전파하면 '대략적 해부학 위치' 임베딩이 된다.)
VESSEL_TERRITORY: dict[int, tuple[int, int]] = {
    1:  (1,  0),   # BA
    2:  (2, +1),   # R-P1P2
    3:  (2, -1),   # L-P1P2
    4:  (3, +1),   # R-ICA-C6-C7
    5:  (5, +1),   # R-M1
    6:  (3, -1),   # L-ICA-C6-C7
    7:  (5, -1),   # L-M1
    8:  (3, +1),   # R-Pcom  (ICA 기원)
    9:  (3, -1),   # L-Pcom
    10: (4,  0),   # Acom
    11: (4, +1),   # R-A1A2
    12: (4, -1),   # L-A1A2
    13: (4, +1),   # R-A3
    14: (4, -1),   # L-A3
    15: (4,  0),   # 3rd-A2 (azygos)
    16: (4,  0),   # 3rd-A3
    17: (5, +1),   # R-M2
    18: (5, +1),   # R-M3
    19: (5, -1),   # L-M2
    20: (5, -1),   # L-M3
    21: (2, +1),   # R-P3P4
    22: (2, -1),   # L-P3P4
    23: (1, +1),   # R-VA
    24: (1, -1),   # L-VA
    25: (1, +1),   # R-SCA
    26: (1, -1),   # L-SCA
    27: (1, +1),   # R-AICA
    28: (1, -1),   # L-AICA
    29: (1, +1),   # R-PICA
    30: (1, -1),   # L-PICA
    31: (3, +1),   # R-AChA (ICA 기원)
    32: (3, -1),   # L-AChA
    33: (3, +1),   # R-OA   (ICA 기원)
    34: (3, -1),   # L-OA
    35: (3, +1),   # R-ICA-C1-C5
    36: (3, -1),   # L-ICA-C1-C5
}
N_TERRITORY = 5


# ---------------------------------------------------------------------------
# 파생 입력 채널 구성 (Stage A)
# ---------------------------------------------------------------------------
# 각 원소: (feature 이름, nnU-Net 정규화 스킴 키)
#   'zscore' = case 별 z-score, 'nonorm' = 그대로 통과(이미 [0,1]/부호 스케일 정규화됨)
# nnU-Net dataset.json 의 channel_names 값에는 정규화 키를 넣는다(값으로 스킴을 조회하므로).
# 사람이 읽을 채널 의미는 channel_layout.json 로 따로 저장한다.
# ---------------------------------------------------------------------------
# ch0 뇌 마스크 기반 정규화 (센터 간 이질성 대응)
# ---------------------------------------------------------------------------
# 문제: 케이스 전체 볼륨에서 mean/std 를 잡으면 **뇌 바깥이 통계를 오염**시킨다.
#   신규 센터의 안면 익명화(얼굴을 갈아 없앤 자리), 두개골/금속 아티팩트, 목·공기 영역이
#   센터마다 차지하는 비율이 달라서, 같은 뇌 조직이 센터마다 다른 값으로 정규화된다.
#
# 왜 nnU-Net 커스텀 정규화 스킴으로 못 하는가 (조사 결과):
#   default_preprocessor._normalize 는 `normalizer.run(data[c], seg[0])` 로 부른다.
#   스킴이 받는 건 **자기 채널 하나 + seg[0]** 뿐이라 vessel 채널/EDT 에 접근할 수 없다.
#   게다가 스킴 탐색 경로가 nnunetv2 패키지 안으로 하드코딩돼 있어(trainer 처럼 외부
#   등록 훅이 없다) site-packages 를 건드리지 않고는 클래스를 추가할 수도 없다.
#   -> 그래서 정규화를 **우리 채널 빌더 쪽으로 옮기고**, nnU-Net 에는 ch0 을
#      'nonorm' 으로 선언해 그대로 통과시킨다. 학습·추론이 같은 함수를 쓰므로
#      (features.build_channels) 두 경로가 어긋날 여지가 없다.
#
# 켜는 법: TOPANEU_BRAIN_NORM=1 (기본 꺼짐 — 기존 데이터셋 동작을 바꾸지 않는다)
BRAIN_NORM = os.environ.get("TOPANEU_BRAIN_NORM", "0") == "1"

#: 뇌 마스크 반경. '최근접 vessel 까지 거리 <= R' 을 뇌 실질의 근사로 쓴다.
#: 혈관 트리가 뇌 전체를 덮지는 못하므로 넉넉하게 잡는다(30~40mm).
BRAIN_MASK_RADIUS_MM = float(os.environ.get("TOPANEU_BRAIN_MASK_RADIUS_MM", "35.0"))

#: 뇌 마스크 seed 에서 제외할 vessel 라벨 — 목/안와로 길게 뻗어 나가는 가지들.
#: 이걸 안 빼면 반경 35mm 가 ICA-C1-C5 를 따라 목까지, OA 를 따라 안와/얼굴까지 번져서
#: **정확히 배제하려던 영역(안면 익명화 자국)을 도로 포함**한다.
#: 라벨 번호는 Dataset211(vessel) dataset.json 기준: 23/24=R/L-VA, 35/36=R/L-ICA-C1-C5,
#: 33/34=R/L-OA.
BRAIN_MASK_SEED_EXCLUDE: set[int] = {23, 24, 33, 34, 35, 36}

#: ch0 로버스트 정규화(D215, 2026-08-09). median/(IQR/1.349) + clip[-5,20].
#  근거: [[ct-zscore-domain-gap]] — 현행 per-image z-score 는 CT 의 뼈(+1000)/공기(-1000)가
#  std 를 500대로 부풀려 조영 혈관(~300HU)을 z~0.5 로 누른다. MR 은 z~4~9 라 **같은 필터가
#  modality 에 따라 7~10배 다른 값에서 발화해야 한다.** 동맥류 복셀 z 중앙값 비(CT/MR)는
#  현행 0.10, p1~99.5 클립 후 0.17, 뇌마스크 z-score 0.07 인데 median/IQR 만 1.11 로 정렬된다.
#
#  ★ 적용 지점이 중요하다 — **ROI 크롭 후**다. D215 raw 는 D213 의 크롭된 ch0 에 적용해
#  만들었고(src/build_d215_robustnorm.py), 정규화 통계는 네트워크가 실제로 보는 영역에서
#  잡아야 한다. 크롭 전 전체 볼륨에서 잡으면 통계가 달라진다(README_pipeline.md 의
#  'D202 vs D203 비교의 교란 항': 크롭 여부로 0.21σ 이동 + 7.7% gain 변화).
#
#  ★ 추론도 반드시 켜야 한다. dataset.json 이 'nonorm' 이라 nnU-Net 이 보정해주지 않고,
#  끄면 에러 없이 조용히 분포가 어긋난다(BRAIN_NORM 과 같은 함정).
#
# 켜는 법: TOPANEU_ROBUST_NORM=1 (기본 꺼짐)
ROBUST_NORM = os.environ.get("TOPANEU_ROBUST_NORM", "0") == "1"

if BRAIN_NORM and ROBUST_NORM:
    # 둘 다 ch0 을 자기 방식으로 정규화한다. 같이 켜면 어느 쪽이 이겼는지 알 수 없는
    # 데이터셋이 조용히 만들어지므로 여기서 끊는다(2026-08-10 이중정규화 사고의 교훈).
    raise RuntimeError(
        "TOPANEU_BRAIN_NORM 과 TOPANEU_ROBUST_NORM 은 배타적이다 — 하나만 켜라 "
        "(D214=BRAIN_NORM, D215=ROBUST_NORM)"
    )

# 'zscore' 를 쓰면 nnU-Net 이 전체 볼륨 통계로 정규화한다. 우리가 ch0 을 직접 정규화하는
# 모드(뇌 마스크 / 로버스트)에서는 nnU-Net 이 손대면 안 된다 -> 'nonorm'.
# ★ 이 선언은 dataset.json 에만 반영된다. plans 를 다른 데이터셋에서 승계하면
#   normalization_schemes 가 같이 따라와 이 선언을 덮어쓴다 — 2026-08-10 에 D215 가
#   그렇게 이중정규화됐다. plans 를 승계하는 스크립트는 반드시 스킴을 재작성할 것.
IMAGE_CHANNEL = ("image", "nonorm" if (BRAIN_NORM or ROBUST_NORM) else "zscore")

# 활성 파생 채널. ablation 시 이 리스트만 바꾸면 됨.
ACTIVE_DERIVED_CHANNELS: list[tuple[str, str]] = [
    ("vessel_proximity", "nonorm"),   # 아이디어 1: vessel 근접도 exp(-d/tau)  [0,1]
    ("territory",        "nonorm"),   # 아이디어 1: 최근접 vessel territory /5  [0,1]
    ("laterality",       "nonorm"),   # 아이디어 1: 최근접 vessel 좌/우 부호     {-1,0,1}
    ("modality",         "nonorm"),   # 아이디어 3(채널판): CT=1 / MR=0
    ("symmetry",         "zscore"),   # 아이디어 2: 정중시상면 반전 비대칭
]


def channel_layout() -> list[tuple[str, str]]:
    """전체 채널 순서 [(이름, 정규화키), ...] (0번=image, 이후 파생)."""
    return [IMAGE_CHANNEL] + ACTIVE_DERIVED_CHANNELS


if __name__ == "__main__":
    # 빠른 자기 점검
    cases = list_case_ids()
    split = load_split()
    m = case_to_nnunet_map()
    print(f"cases: {len(cases)}  (ct={sum(is_ct(c) for c in cases)}, mr={sum(not is_ct(c) for c in cases)})")
    print(f"split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"sample map: {cases[0]} -> {m[cases[0]]}")
    print(f"channels: {[c[0] for c in channel_layout()]}")
