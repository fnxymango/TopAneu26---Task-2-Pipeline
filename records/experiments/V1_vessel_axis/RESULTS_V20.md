# V2-0 — ICA C6/C7/terminus 분할 기준

fine GT side 62 · 그중 **볼 수 있는 것(train·val) 42** · test 20 는 열지 않음

## 0. fine GT 가 수작업인지 규칙 산출물인지

각 side 에서 수작업 경계를 t 로 환산한 값. 전부 0.40/0.75 면 규칙으로 찍어낸 것이다.

| split | 케이스 | side | C6 끝 t | C7 끝 t |
|---|---|---|---|---|
| train | topaneu_center1_mr_017 | L | 0.506 | 0.914 |
| train | topaneu_center1_mr_017 | R | 0.509 | 0.893 |
| train | topaneu_center1_mr_033 | L | 0.491 | 0.925 |
| train | topaneu_center1_mr_033 | R | 0.650 | 0.939 |
| val | topaneu_center1_mr_050 | L | 0.565 | 0.869 |
| val | topaneu_center1_mr_050 | R | 0.563 | 0.891 |
| val | topaneu_center1_mr_053 | L | 0.654 | 0.882 |
| val | topaneu_center1_mr_053 | R | 0.541 | 0.877 |
| train | topaneu_center1_mr_064 | L | 0.547 | 0.915 |
| train | topaneu_center1_mr_064 | R | 0.516 | 0.845 |
| train | topaneu_center1_mr_076 | L | 0.587 | 0.926 |
| train | topaneu_center1_mr_076 | R | 0.609 | 0.914 |
| val | topaneu_center1_mr_085 | L | 0.463 | 0.830 |
| val | topaneu_center1_mr_085 | R | 0.537 | 0.840 |
| train | topaneu_center1_mr_088 | L | 0.620 | 0.931 |
| train | topaneu_center1_mr_088 | R | 0.415 | 0.873 |
| train | topaneu_center1_mr_113 | L | 0.615 | 0.865 |
| train | topaneu_center1_mr_113 | R | 0.581 | 0.966 |
| train | topaneu_center1_mr_119 | L | 0.534 | 0.902 |
| train | topaneu_center1_mr_119 | R | 0.475 | 0.814 |
| train | topaneu_center1_mr_702 | L | 0.664 | 0.861 |
| train | topaneu_center1_mr_702 | R | 0.608 | 0.837 |
| train | topaneu_center2_ct_130 | L | 0.601 | 0.856 |
| train | topaneu_center2_ct_130 | R | 0.994 | 0.490 |
| train | topaneu_center2_ct_172 | L | 0.526 | 0.846 |
| train | topaneu_center2_ct_172 | R | 1.000 | 0.642 |
| train | topaneu_center2_ct_174 | L | 0.647 | 0.929 |
| train | topaneu_center2_ct_174 | R | 1.000 | 0.533 |
| val | topaneu_center2_mr_074 | L | 0.551 | 0.863 |
| val | topaneu_center2_mr_074 | R | 1.000 | 0.487 |
| val | topaneu_center2_mr_083 | L | 0.598 | 0.879 |
| val | topaneu_center2_mr_083 | R | 0.534 | 0.879 |
| train | topaneu_center4_ct_002 | L | 0.572 | 0.848 |
| train | topaneu_center4_ct_002 | R | 0.493 | 0.858 |
| val | topaneu_center4_ct_059 | L | 0.534 | 0.880 |
| val | topaneu_center4_ct_059 | R | 0.484 | 1.000 |
| train | topaneu_center4_ct_074 | L | 0.453 | 0.873 |
| train | topaneu_center4_ct_074 | R | 0.706 | 0.843 |
| train | topaneu_center4_ct_083 | L | 0.429 | 0.964 |
| train | topaneu_center4_ct_083 | R | 0.507 | 0.900 |
| train | topaneu_center4_ct_108 | L | 0.552 | 0.870 |
| train | topaneu_center4_ct_108 | R | 0.470 | 0.843 |

C6 끝 t   중앙 0.552  범위 0.415~1.000  (jskim 고정 0.400)
C7 끝 t   중앙 0.873  범위 0.487~1.000  (jskim 고정 0.750)

→ 산포가 있다. 수작업 경계다.

## 1. 분지가 실제로 t 어디에 있나 — train 전수 (라벨 안 봄)

| 분지 | 잡힌 side | t 중앙 | 25~75% | jskim 절단점과 비교 |
|---|---|---|---|---|
| OA (안동맥) | 421/579 = 73% | 0.181 | 0.152~0.215 |  |
| Pcom | 268/579 = 46% | 0.587 | 0.542~0.630 | +0.187 |
| AChA | 312/579 = 54% | 0.738 | 0.694~0.768 | -0.012 |
| M1 (종말) | 571/579 = 99% | 0.904 | 0.889~0.914 | -0.096 |
| A1 (종말) | 556/579 = 96% | 0.845 | 0.813~0.873 | -0.155 |

## 2. 세 분할 비교 (train·val 10케이스에서만)

(C) train 재추정 절단점 = **0.562 / 0.871**  (jskim 0.400/0.750 · 차이 +0.162 / +0.121)

| 분할 | side | C6 Dice | C7 Dice | terminus Dice | 평균 |
|---|---|---|---|---|---|
| A jskim 0.40/0.75 | 42 | 0.818 | 0.715 | 0.731 | **0.755** |
| C train 재추정 | 42 | 0.828 | 0.669 | 0.558 | **0.685** |
| B 분지기준 | 14 | 0.846 | 0.537 | 0.723 | **0.702** |
| B 분지기준(폴백) | 28 | 0.800 | 0.641 | 0.525 | **0.656** |
| **B 전체(분지+폴백)** | 42 | 0.815 | 0.606 | 0.591 | **0.671** |

분지(Pcom·AChA 둘 다)가 잡힌 side 14/42 = 33%

## 3. 판정 (v20_icasplit.py 머리말 규칙)

1. B(0.671) ≥ A(0.755) ? → 아니오
2. B 가 A 보다 0.05 이상 낮은가? → 예 · C 로 간다
3. 분지 커버리지 33% ≥ 40% ? → 아니오 · C 로 간다
4. C 절단점이 0.40/0.75 와 ±0.05 초과 차이? → 예 · 누수 영향 실재, 앞으로 A 사용 금지

**→ V2-A 에서 쓸 분할 규칙: C train 재추정**
