# TopAneu26 — Task 2 Pipeline

[TopAneu-26](https://github.com/FeynmanDNA/TopAneu-26) Task 2 (뇌동맥류 **52클래스 해부학적 위치 분할**)용 추론 파이프라인입니다.
CTA 또는 MRA 영상 한 장을 받아, 동맥류 복셀마다 52개 위치 클래스 중 하나를 붙인 라벨맵을 냅니다.

이 저장소는 우리 실험에서 성능이 가장 좋았던 구성의 **코드 전체**입니다. 들어 있는 것은 Grand Challenge 컨테이너(`/opt/app`)에 들어가는 파일 그대로입니다.
학습된 가중치는 git 트리에 없고 Releases `V1`에 따로 올렸습니다([가중치와 데이터](#가중치와-데이터) 참고). 실험 기록은 [`records/`](records/README.md)에 있습니다.

---

## 전체 파이프라인

```
 입력 CTA / MRA (.mha)                         모달리티는 GC 인터페이스로 결정 (파일명 아님)
        │
        ▼
 ① 강도 정규화      모달리티별 robust z-score   CT: fg = HU > -300 · MR: fg = x > max(1, 0.02·p99.5)
        │                                      → (x − median_fg) / IQR_fg
        ├───────────────────────────────┐
        ▼                               ▼
 ② 동맥류 검출 (nnU-Net) ※         ④ 혈관 분할 (Skeleton Recall nnU-Net)
    Dataset722 · 3클래스               Dataset800 · 36개 혈관 클래스 · 원본 영상 입력
    stock PlainConvUNet 250ep          ResEncUNet-M · ClassWeightedV2 500ep · fold 0
    fold 0,1,2 확률 평균
        │                               │
 ③ 라벨 2(동맥류) 이진화            ⑤ 혈관 후처리 V5
        │                               │   close → 작은 조각 제거 → 인접성 검사 → 끝점 재연결
        │                               ▼
        │                          ⑥ 분기점 그래프 C4
        │                               │   중심선(Lee thinning) → 스퍼 제거 → 혈관 클래스가 바뀌는 지점 = 분기점
        ▼                               │
 ⑦ 검출 필터  ◀─────────────────────────┤
    blob < 12 복셀 제거 ★               │
    혈관에서 1.0 mm 넘게 떨어진 blob 제거 │
        │                               │
        ▼                               ▼
 ⑧ 위치 분류  Random Forest (seed 3)  ◀──┘
    병변마다 피처: 혈관 근접도 36 · 병변 내부 혈관 점유율 36 · 분기점 근접도 · 랜드마크 좌표
    gC: 1·2등 확률이 가까우면 2등 클래스도 조각으로 함께 출력 (top-k = 2)
    2등 조각은 blob 의 35 %를 받음 ★ (TOPANEU_TOPK_FRAC=0.35)
    출력 부피 1.32배 보정 ★ (TOPANEU_OUT_GROW=1.32, 껍질 복셀을 거리순으로 추가 · 라벨은 최근접 상속)
        │
        ▼
 ⑨ 패치 CNN 환각 필터 ※
    남은 blob 마다 4채널 64³ @ 0.5 mm 패치 (영상 · 혈관 이진 · 혈관 36클래스 · territory)
    점수 < 2.0064e-4 이면 제거
        │
        ▼
 출력 52클래스 위치 라벨맵 (.mha · uint8 · 입력과 같은 격자)
```

※ 표시는 공동 작업자가 설계·구현한 구성요소입니다([구성요소별 출처](#구성요소별-출처) 참고).
★ 표시는 제출 컨테이너(2026-09-10) 이후 실험으로 채택된 변경입니다. [제출본과의 차이](#제출본과의-차이)에 정리했습니다.

①–⑧은 서브프로세스 하나(`pipeline_case.py`)로 돌고, ⑨는 별도 서브프로세스로 돕니다.
두 코드가 서로 다른 nnU-Net을 쓰기 때문입니다. ①–⑧은 동봉한 Skeleton-Recall 포크가 필요하고, ⑨는 pip로 설치한 nnunetv2를 씁니다.
그래서 인터프리터 하나에 같이 올리지 않습니다.

---

## 디렉터리 구조와 파일별 역할

```
TopAneu26---Task-2-Pipeline/
├── Dockerfile
├── README.md
├── app/                                   ← 컨테이너의 /opt/app
│   ├── main.py
│   ├── inference.py
│   ├── topaneu_integrated.py
│   ├── fast_stages.py
│   ├── run_patch_filter.py
│   ├── requirements.txt
│   ├── src/                               ← ⑨ 패치 CNN 환각 필터 ※
│   └── topaneu/                           ← ①–⑧ 본체 (번들 루트, ENV TOPANEU_BUNDLE)
│       ├── code/sblee/nnunet/scripts/     ← 단계별 코드
│       ├── code/sblee/nnunet/analysis/    ← RF 피클 자리 (Releases에서 받음)
│       ├── code/TopAneu-26/eval/task2/    ← 조직위 공식 채점 코드 (평가용)
│       ├── dataset/TopAneu/               ← 52클래스 위치 이름표
│       ├── experiments/V5_.../            ← 혈관 후처리 파라미터
│       ├── nnunet/nnUNet_raw/             ← 두 nnU-Net 데이터셋의 dataset.json
│       └── vendor/Skeleton-Recall/        ← nnU-Net v2 포크
└── records/                               ← 실험 기록 (규칙 · 제출 명세 · 실험 80개)
```

### 진입점과 오케스트레이션 — `app/`

| 파일 | 하는 일 |
|---|---|
| `main.py` | Grand Challenge 공식 템플릿 원본이며 수정하지 않았습니다. `/input/inputs.json`을 보고 CT/MR 인터페이스를 고른 뒤, 영상을 읽어 `infer_ct` 또는 `infer_mr`를 부르고 결과를 `/output/images/aneurysm-segmentation/output.mha`로 씁니다. |
| `inference.py` | `infer_ct` / `infer_mr`. 모달리티를 `TOPANEU_MODALITY`에 담아 `topaneu_integrated.predict_location`을 부릅니다. 모달리티를 파일명이 아니라 인터페이스로 정하는 이유가 있습니다. GC 입력 파일명은 익명 UUID라서, 파일명으로 판별하면 컨테이너에서 실패합니다. |
| `topaneu_integrated.py` | 전체 오케스트레이터입니다. ①–⑧을 `pipeline_case.py` 서브프로세스로 돌린 뒤(`PYTHONPATH`에 vendor 포크를 넣음), ⑨를 `run_patch_filter.py` 서브프로세스로 돌립니다(`PYTHONPATH`를 뺌). 단계별 시간과 메모리도 기록합니다. |
| `fast_stages.py` | ⑦ 검출 필터의 고속판(`filter_case_fast`)입니다. 원본은 볼륨 전체에 EDT를 계산하지만, 여기서는 blob 주변만 잘라 계산합니다. 출력은 원본과 같습니다. `crop_to_vessel`(분기점 그래프 고속화)은 구현만 돼 있고 호출되지는 않습니다. |
| `run_patch_filter.py` | ⑨의 단일 케이스 실행기입니다. 위치 라벨맵, 원본 영상, ④의 혈관맵을 받아 `src/patch_filter`를 적용합니다. |
| `requirements.txt` | 검증 환경에 맞춘 고정 버전입니다. torch는 베이스 이미지(2.5.1+cu121)의 것을 쓰므로 목록에 없습니다. nnunetv2도 없는데, 설치하면 vendor 포크를 가리기 때문입니다. `scikit-learn==1.7.2`는 RF를 언피클하려면 정확히 이 버전이어야 합니다. `seaborn`은 vendor 로거가 import하므로 없으면 검출기 로드가 실패합니다. |

### ⑨ 패치 CNN 환각 필터 — `app/src/` (공동 작업자 구현)

| 파일 | 하는 일 |
|---|---|
| `patch_filter.py` | blob마다 4채널 64³ 패치(0.5 mm 등방, 시야 32 mm)를 떼어 작은 3D CNN으로 "진짜 동맥류인가" 점수를 매기고, 임계 미만인 blob을 지웁니다. 임계는 학습 때 정해 `meta.json`에 싣고, 추론 때 다시 고르지 않습니다. |
| `features.py` | 패치 채널을 만듭니다. 뇌 마스크 기반 z-score, 혈관 이진, 혈관 36클래스, 최근접 혈관의 territory를 계산합니다. 학습과 추론이 같은 함수를 씁니다. |
| `config.py` | 혈관 라벨에서 territory·좌우로 가는 매핑 표, 모달리티 판별, 경로 설정입니다. |

### ①–⑧ 본체 — `app/topaneu/code/sblee/nnunet/scripts/`

**추론 경로에서 실제로 호출되는 파일**

| 파일 | 단계 | 하는 일 |
|---|---|---|
| `pipeline_case.py` | ①–⑧ | 단일 케이스 파이프라인 전체입니다. 정규화, 검출기 3폴드 추론, 혈관 추론, 후처리, 분기점, 필터, 분류를 차례로 호출하고 최종 라벨맵을 씁니다. 단계마다 메모리를 해제해 RAM 최대치를 낮춥니다. ★ `min_vox`와 출력 부피 보정이 이 파일에 있습니다. |
| `p_build_3cls.py` | ① | 모달리티별 robust z-score(`robust_z`)입니다. 검출기 학습 데이터셋(Dataset722, 배경/혈관/동맥류 3클래스)을 만드는 스크립트이기도 합니다. 공동 작업자의 검출 단계 레시피(3클래스 라벨·정규화)를 이 코드베이스에 재구현한 것입니다. |
| `det_filter.py` | ⑦ | 검출 필터의 정답 불필요판입니다. 26-연결 blob 중 `min_vox` 미만이거나, 혈관까지 거리가 `max_dist` mm를 넘는 것을 버립니다. 컨테이너에서는 `fast_stages.filter_case_fast`가 같은 일을 더 빨리 합니다. |
| `postprocess_vessel.py` | ⑤ | 혈관 후처리 V5입니다. 클래스별 closing, GT에서 학습한 최소 조각 크기로 prune, 해부학적으로 인접할 수 없는 고아 조각 제거, skeleton 끝점을 A*로 재연결하는 순서로 돕니다. 전역 largest-CC는 쓰지 않습니다. |
| `c4_branchpoint_graph.py` | ⑥ | 혈관 중심선에서 분기점을 뽑습니다. 52개 위치 클래스 중 21개가 "두 혈관이 만나는 지점"으로 정의돼 있어서(VA-PICA, BA tip, ICA-Pcom, Acom…) 분류기의 핵심 근거가 됩니다. |
| `c5_location_v2.py` | ⑧ | 위치 분류기입니다. 병변 피처를 추출하고 RF 확률에 클래스 사전확률을 보정한 뒤 gC 2등 조각을 내고 부피를 보정합니다. `eval`/`build` 하위 명령은 학습·평가에 씁니다. ★ gC 2등 조각 지분(TOPK_FRAC)이 이 파일에 있고, 출력 부피 보정(OUT_GROW)은 `eval` 경로용으로 여기에도 있습니다. |
| `d9xx_lib.py` | ⑧ | 공용 유틸입니다. 공식 52클래스 이름표, 혈관 클래스 이름, split 로딩을 맡습니다. |
| `postproc_params.json` | ⑤ | 후처리 파라미터입니다(클래스별 최소 조각, 인접 가능 클래스). |

**학습·분석용 파일** (추론에는 쓰이지 않지만 재현에 필요합니다)

| 파일 | 하는 일 |
|---|---|
| `c7_detect_postproc.py` | ⑦ 필터의 `min_vox` × `max_dist` 스윕과 병변 단위 민감도·FP/case 평가 |
| `c8_classifier_cv.py` | 분류기 환자 단위 5-fold 교차검증, 피처 블록 ablation, 혼동 쌍 진단 |
| `c10_landmark_coords.py` | BA tip과 좌우 ICA terminus 세 분기점으로 머리 좌표계를 세우고 정규화 좌표 피처를 만듦. 추론에서는 `c5`가 이 계산을 직접 함 |
| `final_infer.py` | 세 중간 산출물(검출·혈관·분기점)과 RF 피클만으로 ⑧을 단독 실행 (sanity 대조용) |
| `mha_nifti.py` | `.mha` ↔ `.nii.gz` 변환과 왕복 검증 (SimpleITK LPS / nibabel RAS 차이 처리) |

### 그 밖의 폴더

| 경로 | 내용 |
|---|---|
| `topaneu/code/TopAneu-26/eval/task2/` | 조직위 공식 채점 코드 사본 (`c5 eval`이 사용) |
| `topaneu/dataset/TopAneu/dataset_split.json` | 52클래스 위치 이름표(`location_classes`)만 남긴 축약본입니다. 원본에 있던 케이스 목록과 메타는 공개 저장소라서 뺐습니다. |
| `topaneu/experiments/V5_vessel_classweighted_postproc_417/postproc_params.json` | ⑥이 읽는 혈관 인접 표 (`scripts/` 안의 것과 같음) |
| `topaneu/nnunet/nnUNet_raw/Dataset{722,800}_*/dataset.json` | 두 nnU-Net 모델의 채널·라벨 정의 |
| `topaneu/vendor/Skeleton-Recall/` | [Skeleton Recall](https://github.com/MIC-DKFZ/Skeleton-Recall)(nnU-Net v2 포크, Apache-2.0)입니다. 추가한 것은 혈관 트레이너 `nnUNetTrainerSkeletonRecallNoMirroringClassWeighted{,V2}_500ep.py`와 정규화 `preprocessing/normalization/topaneu_norm.py`입니다. 문서 이미지는 용량 때문에 뺐습니다. |

---

## 구성요소별 출처

이 파이프라인은 팀 공동 작업 결과를 통합한 것입니다.

| 구성요소 | 출처 |
|---|---|
| ① 정규화 · ② 동맥류 검출기 (Dataset722, 3클래스 설계, robust z-score) | **공동 작업자 설계·구현**. 이 저장소의 `p_build_3cls.py`는 그 레시피를 재구현한 학습 데이터 빌드 코드이고, 학습된 체크포인트도 그 레시피를 따릅니다. |
| ⑨ 패치 CNN 환각 필터 (`app/src/`, `run_patch_filter.py`) | **공동 작업자 설계·구현** |
| ③–⑧ 혈관 분할·후처리, 분기점 그래프, 검출 필터, 위치 분류(RF·gC), ★ 변경 3개, 통합 컨테이너 | 저장소 작성자 |
| `vendor/Skeleton-Recall/` | MIC-DKFZ 원저작물 (Apache-2.0), 혈관 트레이너와 정규화만 추가 |
| `main.py`, `code/TopAneu-26/` | TopAneu-26 조직위 템플릿·채점 코드 |

---

## 성능

**채점 기준**: 조직위 개정 공식 eval(commit `60765a5`)로 채점했습니다. 로컬 held-out에서 test 83케이스, val 42케이스를 썼습니다.
**반복**: 값은 RF 학습 시드 0–9의 평균 ± 표준편차입니다. 컨테이너에는 그중 seed 3 모델이 들어 있습니다.
**비교 기준**: 가장 가까운 이전 구성(제출본 + FRAC 0.35 + OUT_GROW 1.32)과 비교했습니다.

| split | 구성 | MCC | Precision | Recall |
|---|---|---|---|---|
| test | 이전 구성 | 0.5551 ± 0.0185 | 0.5319 ± 0.0161 | 0.5086 ± 0.0192 |
| test | **이 저장소** | **0.5917 ± 0.0198** | **0.5826 ± 0.0169** | 0.5086 ± 0.0192 |
| val | 이전 구성 | 0.6325 ± 0.0173 | 0.5458 ± 0.0192 | 0.6081 ± 0.0224 |
| val | **이 저장소** | **0.6585 ± 0.0178** | **0.5666 ± 0.0197** | 0.6081 ± 0.0224 |

MCC와 Precision은 10개 시드 전부에서 올랐습니다(10/10).
찾아낸 병변 수(TP)는 그대로이고 잘못 낸 예측(FP)만 줄었습니다(test 52.2 → 46.0, val 22.2 → 21.2).

---

## 제출본과의 차이

기준선은 2026-09-10에 Grand Challenge에 제출한 컨테이너입니다. 그 뒤 채택된 변경은 세 개이고, 셋 다 학습된 가중치는 건드리지 않습니다.

| # | 변경 | 위치 | 효과 |
|---|---|---|---|
| 1 | gC 2등 조각에 blob의 35 %를 줌 (이전에는 3 복셀 고정) | `c5_location_v2.py` · `TOPANEU_TOPK_FRAC=0.35` | 클래스 존재 판정은 그대로이고 Dice·VolSim만 오름 (test Dice +0.008, VolSim +0.010) |
| 2 | 출력 부피를 1.32배로 보정 | `pipeline_case.py` (+ `c5_location_v2.py` eval 경로) · `TOPANEU_OUT_GROW=1.32` | 적중 병변의 예측/GT 부피 중앙값이 0.76–0.82라서 역수만큼 키움. 존재 판정은 그대로이고 Dice·VolSim만 오름 |
| 3 | 검출 blob 최소 크기 5 → 12 복셀 | `pipeline_case.py` · `TOPANEU_MIN_VOX=12` | 3폴드 평균 검출기가 남기는 작은 허위 blob을 제거함. 위 성능표의 차이 |

세 변경 모두 ENV로 켜고 끕니다. `TOPANEU_MIN_VOX=5 TOPANEU_TOPK_FRAC=0 TOPANEU_OUT_GROW=0`으로 끄면 제출본과 같은 동작이 됩니다.

---

## 가중치와 데이터

이 저장소의 git 트리에는 **영상, 정답 마스크, 학습된 가중치가 없습니다.** 코드, 설정 파일, 실험 기록만 있습니다.
학습된 가중치는 용량 때문에 [Releases `V1`](https://github.com/fnxymango/TopAneu26---Task-2-Pipeline/releases/tag/V1)에 따로 올렸습니다.

| Release 파일 | 크기 | 내용 | 들어가는 곳 |
|---|---|---|---|
| `topaneu26-task2-model-weights.tar.gz` | 1.8 GB | 검출기 `models/detector/Dataset722_…/fold_{0,1,2}` · 혈관 `models/vessel/Dataset800_…/fold_0` · 패치 필터 `patchclf/{headA.pt, meta.json}` | 풀어서 `/opt/ml/model`에 마운트 |
| `final_rf_seed3.pkl` | 68 MB | 위치 분류 RF (seed 3) | 빌드 전에 `app/topaneu/code/sblee/nnunet/analysis/`에 둠 → 이미지 안 |
| `SHA256SUMS` | | 두 파일의 SHA-256 | |

```bash
V=https://github.com/fnxymango/TopAneu26---Task-2-Pipeline/releases/download/V1
curl -LO $V/topaneu26-task2-model-weights.tar.gz
curl -LO $V/final_rf_seed3.pkl
curl -LO $V/SHA256SUMS && sha256sum -c SHA256SUMS

mkdir -p model && tar xzf topaneu26-task2-model-weights.tar.gz -C model     # → model/models/, model/patchclf/
cp final_rf_seed3.pkl app/topaneu/code/sblee/nnunet/analysis/
```

모델 탈볼은 `/opt/ml/model`에 풀리고, 이미지 안의 `/opt/app/topaneu/models`가 `/opt/ml/model/models`를 가리키는 심볼릭 링크입니다.
모델 탈볼은 2026-09-10 제출본의 Models 슬롯 파일과 같습니다(md5 `35fa1300046ec75f14fabf8a4d25d26b`).

RF를 다시 학습하려면 train split 병변 피처 표(`c5_location_v2.py build` 출력)가 필요합니다. 이 표에는 정답 라벨이 들어 있어서 올리지 않았습니다.

## 실험 기록

[`records/`](records/README.md)에 이 구성에 이르기까지의 실험 기록 원문(실험 80개, 문서 162개)이 있습니다.
채택 규칙, 시간순 주요 결론, 계열별 실험 목록은 [`records/README.md`](records/README.md)에 정리했습니다.

---

## 빌드와 실행

```bash
# 1) RF 피클을 제자리에 둔다
cp final_rf_seed3.pkl app/topaneu/code/sblee/nnunet/analysis/

# 2) 이미지 빌드
docker build -t topaneu26-task2 .

# 3) 로컬 실행 — GC 입력 레이아웃 그대로
#    test/input/inputs.json                              socket slug: head-ct-angiography 또는 head-mr-angiography
#    test/input/images/head-ct-angio/<아무이름>.mha        (MR 이면 images/head-mr-angio/)
#    model/                                              모델 탈볼을 푼 폴더 (models/, patchclf/)
docker run --rm --gpus all --shm-size 4g \
  -v $PWD/test/input:/input:ro -v $PWD/test/output:/output \
  -v $PWD/model:/opt/ml/model:ro \
  topaneu26-task2
# → test/output/images/aneurysm-segmentation/output.mha
```

주요 환경변수는 Dockerfile에 기본값이 있습니다.

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `TOPANEU_DET_FOLDS` | `0,1,2` | 확률 평균에 쓸 검출기 폴드 |
| `TOPANEU_MIN_VOX` | `12` | ⑦ 최소 blob 크기(복셀) |
| `TOPANEU_TOPK_FRAC` | `0.35` | ⑧ gC 2등 조각 지분 |
| `TOPANEU_OUT_GROW` | `1.32` | ⑧ 출력 부피 배수 |
| `TOPANEU_PATCHCLF` | `/opt/ml/model/patchclf` | ⑨ 패치 필터 폴더. 폴더가 없으면 필터가 꺼짐 |
| `TOPANEU_PF_THRESH` | (meta.json 값) | ⑨ 임계 덮어쓰기 |
| `TOPANEU_TIMING` | `1` | 단계별 소요 시간 출력 |

**Grand Challenge 업로드 형식**: `docker save`가 OCI 레이아웃(`index.json`)을 만들면 GC가 `Could not find manifest.json`이라며 거부합니다.
`tar tzf <image>.tar.gz | head -1`의 결과가 `manifest.json`인 classic docker-archive 형식이어야 합니다.

**자원**: RTX A5000에서 케이스당 최대 RAM 8.2 GB, VRAM 14.7 GB를 썼습니다(test + val 전수 측정).
GC 한도는 케이스당 720초, RAM 31 GB, T4 16 GB입니다.

### 검증

이 저장소 그대로 이미지를 빌드해 제출 컨테이너와 같은 입력(CT 1건, MR 1건)으로 돌렸습니다(2026-09-21, RTX A5000, 케이스당 75–92초).

| 확인 | CT | MR |
|---|---|---|
| ★ 변경 3개를 ENV로 끄면 제출 컨테이너 출력과 같은가 | 복셀 단위 완전 일치 | 복셀 단위 완전 일치 |
| 켜면 출력 라벨 **종류**는 그대로인가 | 같음 (28, 29, 30, 31) | 같음 (29, 31, 50) |
| 켜면 총 부피가 1.32배가 되는가 | 1825 → 2409 | 1403 → 1852 |

---

## 라이선스

이 저장소의 코드는 [MIT](LICENSE)입니다.
단 `app/topaneu/vendor/Skeleton-Recall/`은 원저작물(MIC-DKFZ)의 [Apache-2.0](app/topaneu/vendor/Skeleton-Recall/LICENSE)을 따르고, `app/main.py`와 `app/topaneu/code/TopAneu-26/`은 TopAneu-26 조직위가 배포한 템플릿·채점 코드입니다.
