# TopAneu-26 Task 2 — 최종 제출본 (2026-09-10 기준)

## ① 제출하는 것 — 이 두 파일이 전부다

| 파일 | 크기 | md5 | GC 업로드 위치 |
|---|---|---|---|
| `~/SUBMISSION/final-model-sblee_2026-09-08_15-09-29.tar.gz` | 3.2G | `b9ae77e05d9e466c159ac2566f22f954` | Algorithm → **Containers** |
| `~/SUBMISSION/topaneu-26-task2-integrated-model.tar.gz` | 1.8G | `35fa1300046ec75f14fabf8a4d25d26b` | Algorithm → **Models** |

> 두 슬롯을 바꿔 올리면 `Could not find manifest.json` 으로 임포트가 실패한다.
> 컨테이너 tar 은 classic docker-save 형식(manifest.json 이 첫 엔트리)이고, 모델 tar 은
> `/opt/ml/model` 에 풀리는 가중치 묶음이다. 2026-09-05 의 임포트 실패가 정확히 이 실수였다.

무결성 재확인:
```bash
cd ~/SUBMISSION && md5sum -c <<'EOF'
b9ae77e05d9e466c159ac2566f22f954  final-model-sblee_2026-09-08_15-09-29.tar.gz
35fa1300046ec75f14fabf8a4d25d26b  topaneu-26-task2-integrated-model.tar.gz
EOF
```

## 무엇이 들어 있나

컨테이너 이미지 `final-model-sblee:latest` (2026-09-09 00:09 KST 빌드, entrypoint `python main.py`).
우리 파이프라인 코드·vendor Skeleton-Recall·RF 분류기가 이미지 안에 구워져 있고,
가중치만 모델 tar 에서 `/opt/ml/model` 로 마운트된다.

```
1 robust z (모달리티별)        ← 모달리티는 GC 인터페이스에서 받는다 (파일명 아님)
2 검출  Dataset722 · stock nnUNetPlans PlainConvUNet 250ep · 5폴드 중 fold 0,1,2 만 사용
3 라벨2 이진화
4 혈관  Dataset800 · SkeletonRecall ClassWeightedV2 500ep fold0   ← 우리 것, md5 30798985…
5 혈관 후처리 V5
6 분기점 그래프 C4
7 검출 필터 (min_vox 5 · max_dist 1.0mm)   ← 크롭 EDT 고속판, 출력 동일
8 위치 분류 RF seed3 (구 피처 e11_feat_hyb_ov) · gC ON (topk n=2)
9 패치 CNN 환각 필터 (jslee) · 4채널 64³@0.5mm · 임계 2.0064e-4
```

주요 환경변수: `TOPANEU_DET_FOLDS=0,1,2` · `TOPANEU_TIMING=1` ·
`TOPANEU_PF_THRESH`(임계 덮어쓰기) · `TOPANEU_PATCHCLF`(디렉토리 없으면 필터 꺼짐).

**우리 최종모델(E9)이 아니다.** 통합본이 쓰는 검출기와 분류기는 8월 28일판이다.
E9 10폴드 ResEncL 과 개정판 RF 는 반영되지 않았고, 아래 ② 에 실험용으로 보관돼 있다.

## ② 추가 실험본 — 통합본에 안 들어간 우리 기여

| 위치 | 내용 |
|---|---|
| `~/e9_bundle_experiment/` | E9 검출기 10폴드 ResEncL(4.6G) · 개정판 RF `final_rf_e9_gcoff_seed3.pkl` · 개정판 피처 `*_NEW.json` · sanity 케이스 · MANIFEST md5 |
| `…/experiments/_c1_realpred/` | 혈관예측 `vespp_{train,val,test}` · 검출결과 `aneu_*` 10종. **분류기 단계 실험은 추론을 다시 안 돌리고 여기서 재개할 수 있다** |
| `…/experiments/_c4_bpgraph/` | 분기점 그래프 |
| `…/experiments/D1_newdata/` | `neweval.py`(개정 공식 채점기) · `POSTMORTEM.md` · 클래스별 실패 진단 |
| `…/dataset/TopAneu/` | GT 마스크 전체 · split · **원본 영상은 test 83 + val 41 만** |
| `~/TopAneu-26/` | 조직위 공식 eval |

train 291 케이스의 원본 영상은 지웠다(14G). 검출기·혈관 **재학습**이 필요해지면
SwitchDrive 개정판을 다시 받아야 한다. 분류기 재학습은 `vespp_train` 이 남아 있어 그대로 가능하다.

## 다음에 해볼 만한 것

1. **E9 + 패치 필터.** 두 기여가 겹치지 않으므로 더해질 수 있다. 걸림돌은 런타임뿐이다.
2. **런타임 재측정.** 통합본 실측 312.9s / 제한 420s 인데 어느 GPU 인지 불명이다.
   우리 sanity 의 로컬→T4 배율 1.5~1.7배를 적용하면 초과한다.
3. **분기점 그래프 고속화.** `fast_stages.crop_to_vessel` 이 구현돼 있는데 호출되지 않는다 —
   31.2s 가 그냥 놀고 있고, 그게 2번 문제의 여유가 된다.
4. **3폴드 재튜닝.** RF 와 c7 임계(min_vox·max_dist)는 5폴드 확률평균 분포에서 정해진 값이다.

정리 내역: `…/experiments/D1_newdata/CLEANUP_2026-09-10.{sh,log}`
