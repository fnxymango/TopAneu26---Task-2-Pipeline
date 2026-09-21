# D600 혈관 seg — 검증 결과 (canonical val 15)

> 모델: ResEnc-M + SkelRecall + NoMirroring, 250ep (Dataset600_TopAneuVessel, 36-class, canonical fold0). env=sbaneu2.
> 검증=메모리안전 predict(-nps1) 후 GT 대비 채점. (학습 자체 val은 36클래스 확률맵 export RAM 스파이크로 crash → 별도 재실행)
> ⚠️ **이 실험은 `batch_dice=True`로 학습되어 7개 클래스가 학습 중 붕괴했다.** 원인·해결은 아래 §해석 참조.
> 후속 실험 `D600_vessel_skelrec_resencm_250ep_bd0`(batch_dice=False)에서 P3P4·L-Pcom 회복.

## 요약
- foreground mean Dice: **0.679** (36클래스 평균)
- mean clDice (binary 혈관트리 연결성): **0.8947** (케이스별 0.855~0.922, MR·CT 안정)
- **0점 7개를 제외한 29클래스 평균: 0.843** — 0.679와의 차이는 성능이 아니라 붕괴한 클래스 때문
> 주요 대혈관은 0.89~0.97로 매우 우수.

## 클래스별 Dice
- 대혈관 0.89~0.97: L/R-ICA(C1-C5 0.965/0.948, C6-C7 0.957/0.944), BA 0.949, R-VA 0.921, M1 0.90, M2 0.89, A1A2 0.89, P1P2 0.89
- 중간 0.52~0.69: L-PICA 0.690, L-AICA 0.660, Acom 0.523
- 0.000: L-Pcom, 3rd-A2, 3rd-A3, R/L-P3P4, R/L-AChA
- 상세: val_per_class_dice.json / val_cldice.json

## 해석 — 0점 7개는 "가늘고 드문 혈관"이 아니다

초기 해석은 0점 클래스를 미세·희귀 혈관 탓으로 봤으나, 실측 결과 **P3P4에는 해당하지 않는다.**

| 클래스 | train 케이스 | train voxel | 부피 순위 | val Dice |
|---|---|---|---|---|
| R-P3P4 | **69/69** | 261,632 | 18/36 | 0.000 |
| L-P3P4 | **69/69** | 228,625 | 19/36 | 0.000 |
| L-PICA | 65/69 | 223,637 | 20/36 | 0.690 |
| R-Pcom | 41/69 | 33,401 | 30/36 | 0.723 |
| L-Pcom | 45/69 | 26,961 | 31/36 | 0.000 |

P3P4는 전 케이스에 존재하고 부피가 PICA·A3·SCA·OA보다 큰데 0점이다. L-Pcom은 R-Pcom보다 케이스가 많은데 0점이다. **빈도·크기로 설명되지 않는다.**

### 실제 원인: epoch 65의 학습 붕괴
학습 로그의 epoch별 pseudo-Dice 추적 결과:

- R-P3P4는 ep41~64 동안 살아 있었고 **ep55에 0.634**까지 도달, L-P3P4는 ep61에 0.559
- **ep65부터 두 클래스가 영구히 0**으로 떨어져 250ep까지 회복 못 함
- 살아있는 전경 클래스 수: ep0 1개 → **ep54에 32개로 정점** → ep70에 27개로 급락 → 최종 29개
- 붕괴 구간이 전체 loss 최대 개선 구간과 일치(train loss ep60 −0.657 → ep70 −0.764)
  → **모델이 소수 클래스를 버리는 대가로 전체 점수를 올린 것**

배제한 원인(모두 실측):
- GT 라벨 오류 — 69케이스 전수 검사에서 좌우 뒤바뀜 0건, 위치 산포도 정상 클래스와 동일
- 샘플링 누락 — `class_locations`에 P3P4 좌표 케이스당 수천 개 존재, 균등 추첨
- 영상에서 안 보임 — 정규화 후 밝기(train 14케이스 표본 중앙값) 배경 0.19 대비 P3P4 2.94~3.18. 더 어두운 R-AICA 2.58은 0.750으로 학습됨
- 출력층 붕괴 — 죽은 채널 weight norm 1.0~2.4로 0이 아님(살아있는 채널 2.8~3.6)

### 원인 설정: `batch_dice=True` + `batch_size=2`
Dice를 배치 단위로 뭉쳐 계산하는데 배치가 2장뿐이라, 특정 클래스가 없는 배치에서는 gradient가 사라지고 있는 배치에서도 큰 혈관에 희석된다. 예측을 비우면 Dice 항이 상수 페널티로 고정되는데 옵티마이저가 이 거래를 받아들였다.

**검증**: `batch_dice=False`로 동일 조건 재학습(`..._250ep_bd0`) → R-P3P4 0.803, L-P3P4 0.697, L-Pcom 0.774으로 회복. 0점 클래스 7개→4개, mean Dice 0.679→0.717, clDice 0.8947→0.9210.
단 5-fold의 학습 pseudo-Dice(패치 proxy)로 보면 fold2·3에서는 P3P4가 여전히 0 → **완전한 해결이 아니라 성공 확률을 올린 것**. fold별 실제 val Dice는 CV 채점 결과(`cv_raw.json`/`cv_pp.json`) 참조.

## 다음
- 동맥류 호발 혈관(ICA·MCA·ACA·BA) 0.89~0.97 → 위치(location) 할당 토대로 충분
- 남은 약점: AChA — 학습 pseudo-Dice 기준 R-AChA는 5개 fold 전부 0, L-AChA는 fold4(0.592)를 뺀 4개 fold에서 0. 3rd-A2/A3는 각 12·18케이스뿐인 azygos ACA 변이
- 개선: 1000ep·5-fold 앙상블, batch_size 상향, 희귀클래스 오버샘플링, 공식 topbrain25_eval(Betti-0/HD95)
