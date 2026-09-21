# 제출본 — TopAneu-26 Task 2 (2026-09-10 확정)

이 폴더의 두 파일이 **제출본 전부**다. 다른 어디에도 제출본 사본은 없다.

| 파일 | 크기 | md5 | GC 업로드 슬롯 |
|---|---|---|---|
| `final-model-sblee_2026-09-08_15-09-29.tar.gz` | 3.2G | `b9ae77e05d9e466c159ac2566f22f954` | Algorithm ▸ **Containers** |
| `topaneu-26-task2-integrated-model.tar.gz` | 1.8G | `35fa1300046ec75f14fabf8a4d25d26b` | Algorithm ▸ **Models** |

> 슬롯을 바꿔 올리면 `Could not find manifest.json` 으로 임포트가 실패한다 (2026-09-05 실측).
> 컨테이너 tar 은 classic docker-save 형식(manifest.json 이 첫 엔트리), 모델 tar 은 `/opt/ml/model` 에 풀리는 가중치다.

무결성 확인:
```bash
cd ~/SUBMISSION && md5sum -c <<'X'
b9ae77e05d9e466c159ac2566f22f954  final-model-sblee_2026-09-08_15-09-29.tar.gz
35fa1300046ec75f14fabf8a4d25d26b  topaneu-26-task2-integrated-model.tar.gz
X
```

## 무엇이 들어 있나

컨테이너 이미지 `final-model-sblee:latest` (2026-09-09 00:09 KST 빌드, entrypoint `python main.py`).

```
1 robust z (모달리티별)        ← 모달리티는 GC 인터페이스에서 받는다 (파일명 아님)
2 검출  Dataset722 · stock nnUNetPlans PlainConvUNet 250ep · 5폴드 중 fold 0,1,2 만 사용
3 라벨2 이진화
4 혈관  Dataset800 · SkeletonRecall ClassWeightedV2 500ep fold0   ← 우리 것, md5 30798985…
5 혈관 후처리 V5
6 분기점 그래프 C4
7 검출 필터 (min_vox 5 · max_dist 1.0mm) · 크롭 EDT 고속판(출력 동일)
8 위치 분류 RF seed3 · 구 피처 e11_feat_hyb_ov · gC ON (topk n=2)
9 패치 CNN 환각 필터 (jslee) · 4채널 64³@0.5mm · 임계 2.0064e-4
```

**우리 기여 중 들어간 것**: 혈관 모델, 파이프라인 코드 13개 중 12개(바이트 동일), RF 분류기.
**들어가지 않은 것**: E9 ResEncL 10폴드 검출기, 개정판 피처 RF, gC OFF → `~/e9_bundle_experiment/` 참조.

## 측정된 성능 (5시드 평균 · 우리 test 83 / val 41)

b1on_pf 근사 기준 (통합본은 검출기 3폴드, 근사는 5폴드라 실물은 이보다 약간 낮을 것이다):

| | test | val |
|---|---|---|
| 신 eval MCC (class-avg) | 0.5725 | 0.6575 |
| covered_gt MCC | 0.4335 | 0.5003 |
| 검출 커버리지 | \-- | 107/130 = 82.3% |
| 검출된 병변 분류 정답률 | \-- | 69.0% |

## 미해결 확인사항

1. 코드 주석의 런타임 실측 312.9s 가 **어느 GPU** 인지 불명. T4 배율 1.5~1.7배를 적용하면 420초 한도를 넘긴다.
2. `fast_stages.crop_to_vessel`(분기점 그래프 31.2초 절감)이 구현돼 있으나 **호출되지 않는다**.
3. RF·c7 임계는 5폴드 확률평균 분포에서 튜닝됐는데 실제로는 3폴드로 돈다.
4. `patch_filter.py` 주석은 학습 혈관맵이 "250ep farm"이라는데 실제 stage-1 은 500ep 이다.
