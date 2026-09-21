# TopAneu-26 Task 2 최종모델 번들 — X5 + gC

**목적**: 이 번들만으로 도커 이미지를 만들어 GrandChallenge 에 올리면, 우리 실험 결과가 그대로 재현된다.
**최종 성적** (test 83 · covered_gt ÷36): **MCC 0.4121** · P 0.3963 · R 0.4646 · Dice 0.2382 · VolSim 0.2652 · HD95 0.6348
official ÷52: MCC 0.2853. val 42: cov.MCC 0.4746 / official 0.3012. (분류기 시드 3 고정)

---

## 0. 가장 먼저 읽을 것 — 실제로 걸렸던 함정 5가지

이 번들을 만들면서 **직접 돌려보고** 잡은 것들이다. 하나라도 어기면 컨테이너가 죽거나 결과가 달라진다.

| # | 함정 | 결과 | 대응 |
|---|---|---|---|
| 1 | 혈관 체크포인트 `checkpoint_final` | 우리 결과와 다른 혈관이 나옴 (md5 상이) | **`checkpoint_best`** 를 쓴다 (번들에 이미 best 만 들어있음) |
| 2 | 모달리티를 **파일명**의 `_ct_`/`_mr_` 로 판정 | GC 파일명은 임의 → `ValueError` 로 즉사 | `pipeline_case.py --modality CT\|MR` 로 **인터페이스에서 받아 넘긴다** |
| 3 | `vendor/Skeleton-Recall` 누락 | 혈관 체크포인트 **로드 자체가 실패**<br>`RuntimeError: Could not find requested nnunet trainer` | vendor 를 `PYTHONPATH` 에 올린다. **추론에도 필수** (재학습과 무관) |
| 4 | 경로 레이아웃 변경 | `postproc_params.json` 등을 못 찾음 | 번들 디렉토리 구조를 **그대로** 유지 (스크립트들이 하드코딩) |
| 5 | torch/sklearn 버전 변경 | 상류 추론 미세 변화 · 피클 로드 실패 위험 | `docker/requirements.txt` 의 핀 고정 버전 사용 |

> GC 템플릿 기본 베이스 이미지는 `pytorch:2.9.1-cuda12.6` 이다. 우리 실험은 **torch 2.5.1+cu121** 이라
> `docker/Dockerfile` 에서 베이스를 교체해 두었다.

---

## 1. 파이프라인 (8단계)

```
원본 영상 (CT 또는 MR, 1케이스)
  1) robust z          모달리티별. CT: fg = x > -300 / MR: fg = x > max(1, 0.02·p99.5)
                       (x − median_fg) / IQR_fg      ※ Dataset722 는 channel_names=noNorm
  2) 동맥류 검출        Dataset722 · nnUNetTrainer_250epochs · 5폴드 checkpoint_best 확률평균 · --disable_tta
  3) 라벨 2 이진화      (0=bg, 1=vessel, 2=aneurysm)
  4) 혈관 분할          Dataset800 · ...ClassWeightedV2_500ep · fold0 checkpoint_best · **원본 영상 입력**
  5) 혈관 후처리        V5: close → prune → adj → endpoint
  6) 분기점 그래프      C4 (Lee thinning → 2mm 스퍼 제거 → 전이점 노드화)
  7) 검출 필터          min_vox = 5 · max_dist = 1.0mm   ← GT 불요판 det_filter.py
  8) 위치 분류          RandomForest(seed 3, 500그루) + β=0.5 확신게이트 τ=0.5
                       + gC: 1등이 ICA(3.x) 이고 p2/p1 > 0.7 이면 blob 중심 3복셀에 2등 클래스
  → 52클래스 위치 라벨맵
```

**gC(2등 조각)가 하는 일**: 공식 지표는 케이스·클래스당 0/1 존재 플래그이고 TP 판정이 `intersection > 0` 이다.
즉 한 복셀만 겹쳐도 TP 이므로, 1등 라벨은 그대로 두고 2등 클래스에 3복셀짜리 조각만 얹는다.
단 아무 데나 얹으면 FP 라, **1등이 ICA 이고 1·2등이 팽팽할 때만** 얹는다 (그 구간 2등 적중률 0.412 vs 손익분기 0.102).

---

## 2. 디렉토리 구조 — **바꾸지 말 것**

```
models/detector/Dataset722_TopAneuPjh3cls417/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
    plans.json · dataset.json · fold_{0..4}/checkpoint_best.pth      (1.2 GB)
models/vessel/Dataset800_TopAneuVessel417/nnUNetTrainerSkeletonRecall...V2_500ep__.../
    plans.json · dataset.json · fold_0/checkpoint_best.pth           (780 MB)
vendor/Skeleton-Recall/          nnUNet 포크 — 혈관 트레이너가 여기 산다. 추론 필수
code/sblee/nnunet/scripts/       pipeline_case.py(진입점) · det_filter.py · postprocess_vessel.py
                                 c4_branchpoint_graph.py · c5_location_v2.py · d9xx_lib.py
                                 postproc_params.json · mha_nifti.py
code/sblee/nnunet/analysis/      final_rf_seed3.pkl (분류기+gC 설정, 67MB) · e11_feat_hyb_ov.json
dataset/TopAneu/                 dataset_split.json  (52클래스 이름표 — 클래스ID 매핑에 필요)
nnunet/nnUNet_raw/Dataset*/      dataset.json (혈관 36클래스 이름표)
experiments/V5_.../              postproc_params.json (혈관 후처리 파라미터)
docker/                          참조 Dockerfile · requirements.txt · inference.py · main.py
sanity/                          로컬 스모크 테스트용 4케이스 (원본 + 단계별 기대출력)
env/                             pip freeze · 버전 기록
MANIFEST.txt                     파일 목록 + 체크포인트 md5
```

---

## 3. 도커 만들기

GC 는 **이미지**와 **Model 탈볼**을 따로 받는다. 용량이 큰 가중치는 탈볼로 보낸다.

```bash
# (a) 이미지에 넣을 것: 코드 + vendor + 메타 (models/ 제외)
mkdir -p build/topaneu
rsync -a --exclude models/ --exclude sanity/ --exclude docker/ <번들>/ build/topaneu/
cp <번들>/docker/{Dockerfile,requirements.txt,inference.py,main.py} build/
cd build && docker build --platform=linux/amd64 -t topaneu-26-task2 .

# (b) Model 탈볼: 가중치만
tar -czf model.tar.gz -C <번들> models
#   GC Algorithm → Models 에 업로드. 컨테이너 안에서 /opt/ml/model/models/... 로 마운트된다.
#   ※ inference.py 의 MODEL_DIR 을 실제 마운트 경로에 맞게 확인할 것.
```

로컬 확인은 GC 템플릿 스크립트를 그대로 쓴다 (`docker/do_test_run.sh`).

---

## 4. 재현 검증 — 이미 통과한 것

`sanity/` 의 4케이스로 **원본 영상 → 최종 라벨맵**을 돌려 기준 출력과 복셀 단위로 대조했다.

```bash
python code/sblee/nnunet/scripts/pipeline_case.py \
    --image sanity/input_raw/topaneu_center2_ct_192_0000.nii.gz \
    --modality CT --out /tmp/out.nii.gz --bundle <번들> --device cuda
# 기대: 병변 2 · 라벨 [28, 29, 30, 31] · 크기 {28:792, 29:3, 30:3, 31:158}
#       sanity/expected/stage3_location/topaneu_center2_ct_192.nii.gz 와 **완전일치**
#       (29·30 의 3복셀이 gC 2등 조각)
```

파일명을 `ANON_XYZ.nii.gz` 로 바꿔서 돌려도 동일 — 모달리티는 인자로만 결정된다.

**측정된 재현 정확도**
| 항목 | 결과 |
|---|---|
| robust z 정규화 | 최대차 `0.000e+00` |
| 분류 단계 (test 83 + val 42 전수) | **125 / 125 완전일치** |
| 번들 단독 e2e (CT, 익명 파일명) | **완전일치** — 라벨 [28,29,30,31] 크기 {792, 3, 3, 158} |
| 번들 단독 e2e (MR 무병변) | **완전일치** |
| 번들 단독 e2e (MR 병변2개) | 4복셀 차 (1.3e-07) — 라벨·클래스 동일 |
| 그 흔들림의 공식 점수 영향 | **없음** — 1복셀·4복셀 두 케이스 모두 6지표 변화 0개 |
| 상류 nnUNet 추론 | 수 복셀 흔들림 (GPU 부동소수점 · 버전 무관, 같은 환경 재실행에서도 발생) |
| `.mha` ↔ `.nii.gz` 왕복 | affine 최대차 `3e-08` · 복셀 최대차 `6e-05` |

---

## 5. 알려진 제약

- **케이스당 약 2분** (검출 5폴드 ~20초 + 혈관 ~40초 + 후처리 ~20초 + 분기점·분류 ~15초, A6000 기준).
  GC 시간 제한을 확인할 것. 초과하면 검출을 5폴드 → 3폴드로 줄일 수 있으나 성능이 떨어진다.
- **가중치 합계 약 2.0 GB.** GC Model 탈볼 용량 제한을 확인할 것.
- **Task 1(동맥류 검출 마스크)은 포함하지 않았다.** 필요하면 7단계 출력(`det_filter` 결과)을 그대로 쓰면 된다.
- 혈관 분할은 **fold 0 단일**이다 (시간 제약). 5폴드 확대는 시도하지 않았다.
