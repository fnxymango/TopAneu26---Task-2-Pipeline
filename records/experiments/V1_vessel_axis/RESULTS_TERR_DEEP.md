# 영역별 약점 심층 — 현 기준선(b1Non_pf) + train OOF (2026-09-14 13:20 KST)

설계 근거는 train OOF 에서만 뽑는다. test 목록(§A)은 현황 확인용이고 룰 설계에 쓰지 않는다.

# §A 현 기준선 test·val 오답 (intweak_b1Non_pf)
# b1Non_pf · GT 병변 130 · 검출 107

## MCA — 병변 31 · 미검출 4 · 검출 27 · 시드정답 110/135 (81%)
오답유형: 인접분절 20 · 미할당 5

| split | 케이스 | GT | 지름 | 정답/5 | 예측(시드별) |
|---|---|---|---|---|---|
| test | center2_ct_160 | L-5.3 M1-M2 junction | 3.1 | 0 | L-5.2 M1 early bifurcation×5 |
| test | center2_ct_184 | L-5.3 M1-M2 junction | 5.5 | 0 | L-5.2 M1 early bifurcation×5 |
| test | center4_ct_105 | L-5.3 M1-M2 junction | 3.4 | 0 | 미할당×5 |
| val | center2_ct_169 | R-5.1 M1 trunk | 15.2 | 0 | R-3.7 ICA C7-terminus×5 |
| val | center5_mr_480 | R-5.2 M1 early bifurcation | 4.5 | 0 | R-5.3 M1-M2 junction×5 |

## ACA — 병변 23 · 미검출 3 · 검출 20 · 시드정답 82/100 (82%)
오답유형: 인접분절 13 · 미할당 5

| split | 케이스 | GT | 지름 | 정답/5 | 예측(시드별) |
|---|---|---|---|---|---|
| test | center4_ct_067 | 4.1 Acom complex | 5.3 | 4 | 4.1 Acom complex×4, L-4.2 A1×1 |
| test | center5_mr_484 | L-4.2 A1 | 3.6 | 0 | 4.1 Acom complex×5 |
| test | center2_ct_141 | L-4.4 A3 | 4.5 | 0 | 미할당×5 |
| val | center1_mr_892 | R-4.2 A1 | 4.5 | 3 | R-4.2 A1×3, R-3.7 ICA C7-terminus×2 |
| val | center5_mr_454 | R-4.3 A2 | 6.3 | 0 | 4.1 Acom complex×5 |

## ICA — 병변 56 · 미검출 14 · 검출 42 · 시드정답 140/210 (67%)
오답유형: 인접분절 45 · 같은혈관군 원거리 25

| split | 케이스 | GT | 지름 | 정답/5 | 예측(시드별) |
|---|---|---|---|---|---|
| val | center2_mr_083 | L-3.1 ICA infraclinoid C1-C5 | 3.4 | 0 | L-3.3 ICA C6-nonOA×5 |
| test | center2_ct_105 | L-3.2 ICA C6-OA-junction | 4.1 | 0 | L-3.3 ICA C6-nonOA×5 |
| test | center1_mr_200 | L-3.3 ICA C6-nonOA | 3.4 | 0 | L-3.2 ICA C6-OA-junction×5 |
| val | center1_mr_401 | L-3.3 ICA C6-nonOA | 3.9 | 0 | L-3.2 ICA C6-OA-junction×5 |
| test | center5_mr_469 | L-3.3 ICA C6-nonOA | 5.2 | 1 | L-3.1 ICA infraclinoid C1-C5×3, L-3.2 ICA C6-OA-junction×1, L-3.3 ICA C6-nonOA×1 |
| test | center1_mr_148 | L-3.4 ICA C7-Pcom-junction | 4.7 | 0 | L-3.3 ICA C6-nonOA×5 |
| val | center4_ct_134 | L-3.5 ICA C7-AChA-junction | 3.9 | 0 | L-3.4 ICA C7-Pcom-junction×5 |
| test | center4_ct_104 | L-3.6 ICA C7-nonBranch | 4.1 | 0 | L-3.4 ICA C7-Pcom-junction×5 |
| val | center1_mr_085 | L-3.7 ICA C7-terminus | 2.1 | 0 | L-4.2 A1×5 |
| test | center2_mr_055 | R-3.1 ICA infraclinoid C1-C5 | 3.4 | 0 | R-3.3 ICA C6-nonOA×5 |
| test | center1_mr_024 | R-3.2 ICA C6-OA-junction | 3.8 | 2 | R-3.3 ICA C6-nonOA×3, R-3.2 ICA C6-OA-junction×2 |
| val | center4_ct_115 | R-3.4 ICA C7-Pcom-junction | 9.0 | 3 | R-3.4 ICA C7-Pcom-junction×3, R-3.5 ICA C7-AChA-junction×2 |
| test | center2_ct_192 | R-3.5 ICA C7-AChA-junction | 5.2 | 0 | R-3.4 ICA C7-Pcom-junction×5 |
| test | center5_mr_474 | R-3.5 ICA C7-AChA-junction | 3.8 | 4 | R-3.5 ICA C7-AChA-junction×4, R-3.4 ICA C7-Pcom-junction×1 |
| test | center1_mr_028 | R-3.6 ICA C7-nonBranch | 3.4 | 0 | R-3.5 ICA C7-AChA-junction×3, R-3.4 ICA C7-Pcom-junction×2 |
| test | center1_mr_523 | R-3.6 ICA C7-nonBranch | 3.6 | 0 | R-3.3 ICA C6-nonOA×5 |

## 후순환 — 병변 20 · 미검출 2 · 검출 18 · 시드정답 41/90 (46%)
오답유형: 같은혈관군 원거리 20 · 인접분절 19 · 다른혈관군 5 · 좌우반전 5

| split | 케이스 | GT | 지름 | 정답/5 | 예측(시드별) |
|---|---|---|---|---|---|
| test | center2_mr_080 | 1.10 BA tip | 10.2 | 0 | L-4.2 A1×3, L-3.6 ICA C7-nonBranch×2 |
| val | center5_mr_009 | 1.4 BA trunk | 20.4 | 0 | 1.5 VA-BA junction×4, L-1.3 VA-PICA junction×1 |
| val | center5_mr_023 | 1.5 VA-BA junction | 9.5 | 1 | R-1.3 VA-PICA junction×4, 1.5 VA-BA junction×1 |
| test | center5_mr_081 | L-1.1 VA trunk | 14.7 | 0 | R-1.1 VA trunk×5 |
| test | center1_mr_262 | L-1.3 VA-PICA junction | 3.3 | 0 | L-1.1 VA trunk×5 |
| val | center2_ct_167 | L-1.3 VA-PICA junction | 4.2 | 0 | L-1.1 VA trunk×5 |
| test | center1_mr_523 | L-1.9 BA-SCA junction | 5.2 | 0 | 1.10 BA tip×5 |
| test | center1_mr_877 | R-1.3 VA-PICA junction | 4.3 | 0 | R-1.1 VA trunk×5 |
| val | center5_mr_480 | R-1.3 VA-PICA junction | 4.8 | 0 | 1.5 VA-BA junction×5 |
| test | center4_ct_129 | R-1.9 BA-SCA junction | 6.2 | 0 | 1.10 BA tip×5 |

## 환각(GT에 없는 케이스×클래스, 시드합) 영역별: ICA 165 · 후순환 74 · MCA 45 · ACA 42

# §B train 5겹×5시드 OOF 혼동
## MCA — train 병변 69 · 판정 345 · top1 87% · top2 94% · top3 97% · 좌우만 틀림 0

| 클래스 | 판정 | top1 | top2 | top3 | RF가 1등으로 고른 횟수 |
|---|---|---|---|---|---|
| 5.1 | 15 | 0% | 33% | 67% | 7 |
| 5.2 | 70 | 76% | 100% | 100% | 71 |
| 5.3d | 5 | 0% | 0% | 0% | 0 |
| 5.3j | 255 | 97% | 98% | 100% | 262 |

주요 혼동(GT→예측, 판정수): 5.2→5.3j 10 · 5.1→5.2 10 · 5.3j→5.2 8 · 5.2→5.1 7 · 5.3d→5.3j 5 · 5.1→3.7 5

## ACA — train 병변 48 · 판정 240 · top1 75% · top2 79% · top3 88% · 좌우만 틀림 2

| 클래스 | 판정 | top1 | top2 | top3 | RF가 1등으로 고른 횟수 |
|---|---|---|---|---|---|
| 4.1 | 160 | 97% | 97% | 100% | 163 |
| 4.2 | 20 | 0% | 15% | 70% | 4 |
| 4.3 | 15 | 20% | 20% | 20% | 11 |
| 4.4 | 30 | 73% | 90% | 97% | 40 |
| 4.5 | 15 | 0% | 13% | 33% | 6 |

주요 혼동(GT→예측, 판정수): 4.2→3.7 15 · 4.5→4.4 12 · 4.4→4.5 5 · 4.3→4.4 5 · 4.1→4.3 5 · 4.2→4.1 5 · 4.3→4.2 4 · 4.3→4.1 3

## 후순환 — train 병변 51 · 판정 255 · top1 64% · top2 75% · top3 79% · 좌우만 틀림 0

| 클래스 | 판정 | top1 | top2 | top3 | RF가 1등으로 고른 횟수 |
|---|---|---|---|---|---|
| 1.1 | 90 | 83% | 100% | 100% | 91 |
| 1.10 | 40 | 100% | 100% | 100% | 45 |
| 1.3 | 15 | 0% | 13% | 20% | 14 |
| 1.4 | 55 | 31% | 45% | 60% | 19 |
| 1.5 | 30 | 70% | 80% | 83% | 44 |
| 1.7 | 5 | 0% | 0% | 0% | 0 |
| 1.9 | 10 | 100% | 100% | 100% | 15 |
| 2.1 | 5 | 0% | 0% | 0% | 5 |
| 2.2 | 5 | 0% | 0% | 0% | 0 |

주요 혼동(GT→예측, 판정수): 1.3→1.1 10 · 1.4→1.5 9 · 1.1→1.5 9 · 1.1→1.3 6 · 1.7→1.9 5 · 2.1→1.10 5 · 1.3→1.5 5 · 1.4→2.1 5

## ICA — train 병변 103 · 판정 515 · top1 59% · top2 80% · top3 90% · 좌우만 틀림 0

| 클래스 | 판정 | top1 | top2 | top3 | RF가 1등으로 고른 횟수 |
|---|---|---|---|---|---|
| 3.1 | 135 | 90% | 99% | 100% | 146 |
| 3.2 | 45 | 40% | 62% | 80% | 38 |
| 3.3 | 105 | 50% | 84% | 87% | 88 |
| 3.4 | 100 | 59% | 79% | 94% | 114 |
| 3.5 | 50 | 22% | 56% | 76% | 35 |
| 3.6 | 35 | 9% | 31% | 66% | 26 |
| 3.7 | 45 | 91% | 100% | 100% | 54 |

주요 혼동(GT→예측, 판정수): 3.5→3.4 24 · 3.3→3.2 20 · 3.6→3.4 20 · 3.2→3.3 18 · 3.4→3.5 16 · 3.3→3.1 16 · 3.1→3.3 14 · 3.4→3.6 11

# §C 사전확률(로짓) 보정 스크리닝 — 기각
α 0    top1 70.7% · macro-recall 0.447 · 고침 0 망침 0 · 영역 macro ICA 0.51 · ACA 0.38 · MCA 0.43 · 후순환 0.44
α 0.25 top1 69.1% · macro-recall 0.434 · 고침 15 망침 37 · 영역 macro ICA 0.52 · ACA 0.37 · MCA 0.43 · 후순환 0.40
α 0.5  top1 66.3% · macro-recall 0.416 · 고침 20 망침 79 · 영역 macro ICA 0.48 · ACA 0.37 · MCA 0.42 · 후순환 0.39
α 0.75 top1 61.5% · macro-recall 0.381 · 고침 26 망침 150 · 영역 macro ICA 0.45 · ACA 0.36 · MCA 0.40 · 후순환 0.33
α 1.0  top1 56.4% · macro-recall 0.352 · 고침 33 망침 225 · 영역 macro ICA 0.42 · ACA 0.34 · MCA 0.40 · 후순환 0.28

클래스별 train 병변 수: {'1.1': 18, '1.10': 8, '1.3': 3, '1.4': 11, '1.5': 6, '1.7': 1, '1.9': 2, '2.1': 1, '2.2': 1, '3.1': 27, '3.2': 9, '3.3': 21, '3.4': 20, '3.5': 10, '3.6': 7, '3.7': 9, '4.1': 32, '4.2': 4, '4.3': 3, '4.4': 6, '4.5': 3, '5.1': 3, '5.2': 14, '5.3d': 1, '5.3j': 51}
