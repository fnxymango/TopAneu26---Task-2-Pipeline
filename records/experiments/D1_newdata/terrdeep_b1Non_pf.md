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
