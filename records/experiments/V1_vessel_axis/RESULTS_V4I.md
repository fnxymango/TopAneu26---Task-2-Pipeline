# V4-I — ICA 원위 기하 룰 (예측 혈관 조건 재보정)

판정규칙은 V4I.sh 머리말에 결과 보기 전에 고정. 1차=병변 TP · 안전=새로 틀린 ≤ 새로 맞힌 · 2차=ΔMCC ≥ −0.005

## 0. 룰 발동 내역 (패치필터 전 · c5 단계)

**test** 발동 129건 · 새로 맞힘 50 · 새로 틀림 57 · 둘 다 아님 22

| 시드 | 케이스 | 병변 | RF 1등 → 룰 | GT | 결과 |
|---|---|---|---|---|---|
| s0 | topaneu_center1_mr_018 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s0 | topaneu_center1_mr_024 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.2 ICA C6-OA-junction | ✓ 수정 |
| s0 | topaneu_center1_mr_028 | 1 | R-3.5 ICA C7-AChA-junction → R-3.5 ICA C7-AChA-junction | R-3.6 ICA C7-nonBranch | – |
| s0 | topaneu_center1_mr_056 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s0 | topaneu_center1_mr_092 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | GT 없음(오탐) | – |
| s0 | topaneu_center1_mr_148 | 2 | L-3.3 ICA C6-nonOA → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s0 | topaneu_center1_mr_200 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s0 | topaneu_center1_mr_523 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.6 ICA C7-nonBranch | – |
| s0 | topaneu_center2_ct_105 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.2 ICA C6-OA-junction | – |
| s0 | topaneu_center2_ct_189 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s0 | topaneu_center2_ct_192 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | – |
| s0 | topaneu_center2_ct_192 | 2 | L-3.5 ICA C7-AChA-junction → L-3.5 ICA C7-AChA-junction | GT 없음(오탐) | – |
| s0 | topaneu_center2_mr_038 | 1 | L-3.5 ICA C7-AChA-junction → L-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s0 | topaneu_center2_mr_042 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.3 ICA C6-nonOA | ✓ 수정 |
| s0 | topaneu_center2_mr_055 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.1 ICA infraclinoid C1-C5 | – |
| s0 | topaneu_center2_mr_069 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s0 | topaneu_center2_mr_086 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s0 | topaneu_center4_ct_063 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s0 | topaneu_center4_ct_072 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s0 | topaneu_center4_ct_104 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.6 ICA C7-nonBranch | – |
| s0 | topaneu_center5_mr_007 | 3 | R-3.3 ICA C6-nonOA → R-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s0 | topaneu_center5_mr_466 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s0 | topaneu_center5_mr_469 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s0 | topaneu_center5_mr_474 | 1 | R-3.5 ICA C7-AChA-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | ✗ 파손 |
| s0 | topaneu_center5_mr_481 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s0 | topaneu_center5_mr_492 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s1 | topaneu_center1_mr_018 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s1 | topaneu_center1_mr_024 | 1 | R-3.2 ICA C6-OA-junction → R-3.2 ICA C6-OA-junction | R-3.2 ICA C6-OA-junction | ✓ 수정 |
| s1 | topaneu_center1_mr_028 | 1 | R-3.5 ICA C7-AChA-junction → R-3.5 ICA C7-AChA-junction | R-3.6 ICA C7-nonBranch | – |
| s1 | topaneu_center1_mr_056 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s1 | topaneu_center1_mr_092 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | GT 없음(오탐) | – |
| s1 | topaneu_center1_mr_148 | 2 | L-3.3 ICA C6-nonOA → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s1 | topaneu_center1_mr_200 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s1 | topaneu_center1_mr_523 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.6 ICA C7-nonBranch | – |
| s1 | topaneu_center2_ct_105 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.2 ICA C6-OA-junction | – |
| s1 | topaneu_center2_ct_189 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s1 | topaneu_center2_ct_192 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | – |
| s1 | topaneu_center2_ct_192 | 2 | L-3.5 ICA C7-AChA-junction → L-3.5 ICA C7-AChA-junction | GT 없음(오탐) | – |
| s1 | topaneu_center2_mr_038 | 1 | L-3.5 ICA C7-AChA-junction → L-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s1 | topaneu_center2_mr_042 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.3 ICA C6-nonOA | ✓ 수정 |
| s1 | topaneu_center2_mr_055 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.1 ICA infraclinoid C1-C5 | – |
| s1 | topaneu_center2_mr_069 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s1 | topaneu_center2_mr_080 | 3 | L-3.6 ICA C7-nonBranch → L-3.4 ICA C7-Pcom-junction | 1.10 BA tip | – |
| s1 | topaneu_center2_mr_086 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s1 | topaneu_center4_ct_063 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s1 | topaneu_center4_ct_072 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s1 | topaneu_center4_ct_104 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.6 ICA C7-nonBranch | – |
| s1 | topaneu_center5_mr_007 | 3 | R-3.3 ICA C6-nonOA → R-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s1 | topaneu_center5_mr_466 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s1 | topaneu_center5_mr_474 | 1 | R-3.5 ICA C7-AChA-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | ✗ 파손 |
| s1 | topaneu_center5_mr_481 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s1 | topaneu_center5_mr_492 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s2 | topaneu_center1_mr_018 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s2 | topaneu_center1_mr_024 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.2 ICA C6-OA-junction | ✓ 수정 |
| s2 | topaneu_center1_mr_028 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.5 ICA C7-AChA-junction | R-3.6 ICA C7-nonBranch | – |
| s2 | topaneu_center1_mr_056 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s2 | topaneu_center1_mr_092 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | GT 없음(오탐) | – |
| s2 | topaneu_center1_mr_148 | 2 | L-3.3 ICA C6-nonOA → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s2 | topaneu_center1_mr_200 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s2 | topaneu_center1_mr_523 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.6 ICA C7-nonBranch | – |
| s2 | topaneu_center2_ct_105 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.2 ICA C6-OA-junction | – |
| s2 | topaneu_center2_ct_189 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s2 | topaneu_center2_ct_192 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | – |
| s2 | topaneu_center2_ct_192 | 2 | L-3.5 ICA C7-AChA-junction → L-3.5 ICA C7-AChA-junction | GT 없음(오탐) | – |
| s2 | topaneu_center2_mr_038 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s2 | topaneu_center2_mr_042 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.3 ICA C6-nonOA | ✓ 수정 |
| s2 | topaneu_center2_mr_055 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.1 ICA infraclinoid C1-C5 | – |
| s2 | topaneu_center2_mr_069 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s2 | topaneu_center2_mr_086 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s2 | topaneu_center4_ct_063 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s2 | topaneu_center4_ct_072 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s2 | topaneu_center4_ct_104 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.6 ICA C7-nonBranch | – |
| s2 | topaneu_center5_mr_007 | 3 | R-3.3 ICA C6-nonOA → R-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s2 | topaneu_center5_mr_466 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s2 | topaneu_center5_mr_474 | 1 | R-3.5 ICA C7-AChA-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | ✗ 파손 |
| s2 | topaneu_center5_mr_481 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s2 | topaneu_center5_mr_492 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s3 | topaneu_center1_mr_018 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s3 | topaneu_center1_mr_024 | 1 | R-3.2 ICA C6-OA-junction → R-3.2 ICA C6-OA-junction | R-3.2 ICA C6-OA-junction | ✓ 수정 |
| s3 | topaneu_center1_mr_028 | 1 | R-3.5 ICA C7-AChA-junction → R-3.5 ICA C7-AChA-junction | R-3.6 ICA C7-nonBranch | – |
| s3 | topaneu_center1_mr_056 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s3 | topaneu_center1_mr_092 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | GT 없음(오탐) | – |
| s3 | topaneu_center1_mr_148 | 2 | L-3.3 ICA C6-nonOA → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s3 | topaneu_center1_mr_200 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s3 | topaneu_center1_mr_523 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.6 ICA C7-nonBranch | – |
| s3 | topaneu_center2_ct_105 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.2 ICA C6-OA-junction | – |
| s3 | topaneu_center2_ct_189 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s3 | topaneu_center2_ct_192 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | – |
| s3 | topaneu_center2_ct_192 | 2 | L-3.5 ICA C7-AChA-junction → L-3.5 ICA C7-AChA-junction | GT 없음(오탐) | – |
| s3 | topaneu_center2_mr_038 | 1 | L-3.5 ICA C7-AChA-junction → L-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s3 | topaneu_center2_mr_042 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.3 ICA C6-nonOA | ✓ 수정 |
| s3 | topaneu_center2_mr_055 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.1 ICA infraclinoid C1-C5 | – |
| s3 | topaneu_center2_mr_069 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s3 | topaneu_center2_mr_086 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s3 | topaneu_center4_ct_063 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s3 | topaneu_center4_ct_072 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s3 | topaneu_center4_ct_104 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.6 ICA C7-nonBranch | – |
| s3 | topaneu_center5_mr_007 | 3 | R-3.3 ICA C6-nonOA → R-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s3 | topaneu_center5_mr_466 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s3 | topaneu_center5_mr_469 | 1 | L-3.3 ICA C6-nonOA → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | ✗ 파손 |
| s3 | topaneu_center5_mr_474 | 1 | R-3.5 ICA C7-AChA-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | ✗ 파손 |
| s3 | topaneu_center5_mr_481 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s3 | topaneu_center5_mr_492 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s4 | topaneu_center1_mr_018 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s4 | topaneu_center1_mr_024 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.2 ICA C6-OA-junction | ✓ 수정 |
| s4 | topaneu_center1_mr_028 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.5 ICA C7-AChA-junction | R-3.6 ICA C7-nonBranch | – |
| s4 | topaneu_center1_mr_056 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s4 | topaneu_center1_mr_092 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | GT 없음(오탐) | – |
| s4 | topaneu_center1_mr_148 | 2 | L-3.3 ICA C6-nonOA → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s4 | topaneu_center1_mr_200 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s4 | topaneu_center1_mr_523 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.6 ICA C7-nonBranch | – |
| s4 | topaneu_center2_ct_105 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.2 ICA C6-OA-junction | – |
| s4 | topaneu_center2_ct_189 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s4 | topaneu_center2_ct_192 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | – |
| s4 | topaneu_center2_ct_192 | 2 | L-3.5 ICA C7-AChA-junction → L-3.5 ICA C7-AChA-junction | GT 없음(오탐) | – |
| s4 | topaneu_center2_mr_038 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s4 | topaneu_center2_mr_042 | 1 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.3 ICA C6-nonOA | ✓ 수정 |
| s4 | topaneu_center2_mr_055 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.1 ICA infraclinoid C1-C5 | – |
| s4 | topaneu_center2_mr_069 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s4 | topaneu_center2_mr_080 | 3 | L-3.6 ICA C7-nonBranch → L-3.4 ICA C7-Pcom-junction | 1.10 BA tip | – |
| s4 | topaneu_center2_mr_086 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s4 | topaneu_center4_ct_063 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s4 | topaneu_center4_ct_072 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s4 | topaneu_center4_ct_104 | 1 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.6 ICA C7-nonBranch | – |
| s4 | topaneu_center5_mr_007 | 3 | R-3.2 ICA C6-OA-junction → R-3.6 ICA C7-nonBranch | GT 없음(오탐) | – |
| s4 | topaneu_center5_mr_466 | 1 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s4 | topaneu_center5_mr_474 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.5 ICA C7-AChA-junction | – |
| s4 | topaneu_center5_mr_481 | 1 | R-3.3 ICA C6-nonOA → R-3.2 ICA C6-OA-junction | R-3.3 ICA C6-nonOA | ✗ 파손 |
| s4 | topaneu_center5_mr_492 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |

**val** 발동 77건 · 새로 맞힘 40 · 새로 틀림 48 · 둘 다 아님 -11

| 시드 | 케이스 | 병변 | RF 1등 → 룰 | GT | 결과 |
|---|---|---|---|---|---|
| s0 | topaneu_center1_mr_085 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s0 | topaneu_center1_mr_085 | 2 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s0 | topaneu_center1_mr_401 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s0 | topaneu_center2_ct_108 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s0 | topaneu_center2_ct_166 | 1 | R-3.2 ICA C6-OA-junction → R-3.3 ICA C6-nonOA | R-3.2 ICA C6-OA-junction | ✗ 파손 |
| s0 | topaneu_center2_ct_169 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-5.1 M1 trunk | – |
| s0 | topaneu_center2_mr_074 | 2 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s0 | topaneu_center2_mr_074 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s0 | topaneu_center2_mr_083 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s0 | topaneu_center2_mr_083 | 2 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.1 ICA infraclinoid C1-C5 | – |
| s0 | topaneu_center2_mr_083 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s0 | topaneu_center4_ct_059 | 1 | R-3.6 ICA C7-nonBranch → R-3.7 ICA C7-terminus | R-3.6 ICA C7-nonBranch | ✗ 파손 |
| s0 | topaneu_center4_ct_115 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s0 | topaneu_center4_ct_134 | 2 | L-3.4 ICA C7-Pcom-junction → L-3.3 ICA C6-nonOA | L-3.4 ICA C7-Pcom-junction | ✗ 파손 |
| s0 | topaneu_center4_ct_134 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.5 ICA C7-AChA-junction | ✓ 수정 |
| s1 | topaneu_center1_mr_085 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s1 | topaneu_center1_mr_085 | 2 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s1 | topaneu_center1_mr_401 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s1 | topaneu_center2_ct_108 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s1 | topaneu_center2_ct_166 | 1 | R-3.2 ICA C6-OA-junction → R-3.3 ICA C6-nonOA | R-3.2 ICA C6-OA-junction | ✗ 파손 |
| s1 | topaneu_center2_ct_169 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-5.1 M1 trunk | – |
| s1 | topaneu_center2_mr_074 | 2 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s1 | topaneu_center2_mr_074 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s1 | topaneu_center2_mr_083 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s1 | topaneu_center2_mr_083 | 2 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.1 ICA infraclinoid C1-C5 | – |
| s1 | topaneu_center2_mr_083 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s1 | topaneu_center4_ct_059 | 1 | R-3.6 ICA C7-nonBranch → R-3.7 ICA C7-terminus | R-3.6 ICA C7-nonBranch | ✗ 파손 |
| s1 | topaneu_center4_ct_115 | 1 | R-3.5 ICA C7-AChA-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s1 | topaneu_center4_ct_134 | 2 | L-3.4 ICA C7-Pcom-junction → L-3.3 ICA C6-nonOA | L-3.4 ICA C7-Pcom-junction | ✗ 파손 |
| s1 | topaneu_center4_ct_134 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.5 ICA C7-AChA-junction | ✓ 수정 |
| s2 | topaneu_center1_mr_085 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s2 | topaneu_center1_mr_085 | 2 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s2 | topaneu_center1_mr_401 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s2 | topaneu_center1_mr_892 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-4.2 A1 | – |
| s2 | topaneu_center2_ct_108 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s2 | topaneu_center2_ct_166 | 1 | R-3.2 ICA C6-OA-junction → R-3.3 ICA C6-nonOA | R-3.2 ICA C6-OA-junction | ✗ 파손 |
| s2 | topaneu_center2_ct_169 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-5.1 M1 trunk | – |
| s2 | topaneu_center2_mr_074 | 2 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s2 | topaneu_center2_mr_074 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s2 | topaneu_center2_mr_083 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s2 | topaneu_center2_mr_083 | 2 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.1 ICA infraclinoid C1-C5 | – |
| s2 | topaneu_center2_mr_083 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s2 | topaneu_center4_ct_059 | 1 | R-3.6 ICA C7-nonBranch → R-3.7 ICA C7-terminus | R-3.6 ICA C7-nonBranch | ✗ 파손 |
| s2 | topaneu_center4_ct_115 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s2 | topaneu_center4_ct_134 | 2 | L-3.4 ICA C7-Pcom-junction → L-3.3 ICA C6-nonOA | L-3.4 ICA C7-Pcom-junction | ✗ 파손 |
| s2 | topaneu_center4_ct_134 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.5 ICA C7-AChA-junction | ✓ 수정 |
| s3 | topaneu_center1_mr_085 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s3 | topaneu_center1_mr_085 | 2 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s3 | topaneu_center1_mr_401 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s3 | topaneu_center2_ct_108 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s3 | topaneu_center2_ct_166 | 1 | R-3.2 ICA C6-OA-junction → R-3.3 ICA C6-nonOA | R-3.2 ICA C6-OA-junction | ✗ 파손 |
| s3 | topaneu_center2_ct_169 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-5.1 M1 trunk | – |
| s3 | topaneu_center2_mr_074 | 2 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s3 | topaneu_center2_mr_074 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s3 | topaneu_center2_mr_083 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s3 | topaneu_center2_mr_083 | 2 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.1 ICA infraclinoid C1-C5 | – |
| s3 | topaneu_center2_mr_083 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s3 | topaneu_center4_ct_059 | 1 | R-3.6 ICA C7-nonBranch → R-3.7 ICA C7-terminus | R-3.6 ICA C7-nonBranch | ✗ 파손 |
| s3 | topaneu_center4_ct_115 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s3 | topaneu_center4_ct_134 | 2 | L-3.4 ICA C7-Pcom-junction → L-3.3 ICA C6-nonOA | L-3.4 ICA C7-Pcom-junction | ✗ 파손 |
| s3 | topaneu_center4_ct_134 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.5 ICA C7-AChA-junction | ✓ 수정 |
| s4 | topaneu_center1_mr_085 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s4 | topaneu_center1_mr_085 | 2 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-3.7 ICA C7-terminus | ✓ 수정 |
| s4 | topaneu_center1_mr_401 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.3 ICA C6-nonOA | – |
| s4 | topaneu_center1_mr_892 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-4.2 A1 | – |
| s4 | topaneu_center2_ct_108 | 1 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | GT 없음(오탐) | – |
| s4 | topaneu_center2_ct_166 | 1 | R-3.2 ICA C6-OA-junction → R-3.3 ICA C6-nonOA | R-3.2 ICA C6-OA-junction | ✗ 파손 |
| s4 | topaneu_center2_ct_169 | 1 | R-3.7 ICA C7-terminus → R-3.7 ICA C7-terminus | R-5.1 M1 trunk | – |
| s4 | topaneu_center2_mr_074 | 2 | R-3.4 ICA C7-Pcom-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s4 | topaneu_center2_mr_074 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.4 ICA C7-Pcom-junction | L-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s4 | topaneu_center2_mr_083 | 1 | R-3.3 ICA C6-nonOA → R-3.3 ICA C6-nonOA | R-3.3 ICA C6-nonOA | ✓ 수정 |
| s4 | topaneu_center2_mr_083 | 2 | L-3.3 ICA C6-nonOA → L-3.3 ICA C6-nonOA | L-3.1 ICA infraclinoid C1-C5 | – |
| s4 | topaneu_center2_mr_083 | 3 | L-3.2 ICA C6-OA-junction → L-3.2 ICA C6-OA-junction | L-3.2 ICA C6-OA-junction | ✓ 수정 |
| s4 | topaneu_center4_ct_059 | 1 | R-3.6 ICA C7-nonBranch → R-3.7 ICA C7-terminus | R-3.6 ICA C7-nonBranch | ✗ 파손 |
| s4 | topaneu_center4_ct_115 | 1 | R-3.5 ICA C7-AChA-junction → R-3.4 ICA C7-Pcom-junction | R-3.4 ICA C7-Pcom-junction | ✓ 수정 |
| s4 | topaneu_center4_ct_134 | 2 | L-3.4 ICA C7-Pcom-junction → L-3.3 ICA C6-nonOA | L-3.4 ICA C7-Pcom-junction | ✗ 파손 |
| s4 | topaneu_center4_ct_134 | 3 | L-3.4 ICA C7-Pcom-junction → L-3.5 ICA C7-AChA-junction | L-3.5 ICA C7-AChA-junction | ✓ 수정 |

## 1. 구현 검증 — 룰이 안 걸린 케이스는 기준선과 같아야 한다

미발동 케이스 동일 444 · 다름 0 ✓

## 2. 1차 · 병변 단위 TP (b1Non_pf → b1v4i_pf)

기준 `b1Non_pf` 392/650 · 후보 `b1v4i_pf` 372/650 · **Δ -20**

| 클래스 | 기준 | 후보 | 기회 | Δ |
|---|---|---|---|---|
| R-3.3 ICA C6-nonOA | 20/20 | 10/20 | 20 | **-10** |
| R-3.2 ICA C6-OA-junction | 10/15 | 5/15 | 15 | **-5** |
| R-3.6 ICA C7-nonBranch | 5/15 | 0/15 | 15 | **-5** |
| R-3.5 ICA C7-AChA-junction | 5/15 | 1/15 | 15 | **-4** |
| L-3.3 ICA C6-nonOA | 9/25 | 8/25 | 25 | **-1** |
| L-3.5 ICA C7-AChA-junction | 0/10 | 5/10 | 10 | **+5** |

새로 맞힌 5 · 새로 틀린 25 · 순 -20

## 3. 2차 · 신 eval 7지표 (b1v4i_pf − b1Non_pf)

# ICA 원위 룰 − 기준선 — 새 공식 eval · b1v4i_pf − b1Non_pf · seed 0~4

## test · 시드 [0, 1, 2, 3, 4]

| 지표 | s0 | s1 | s2 | s3 | s4 | 평균Δ | 개선 |
|---|---|---|---|---|---|---|---|
| PRECISION | -0.0694 | -0.0411 | -0.0300 | -0.0442 | -0.0136 | -0.0397 | × |
| RECALL | -0.0231 | -0.0231 | -0.0231 | -0.0301 | -0.0093 | -0.0218 | × |
| F1 | -0.0186 | -0.0195 | -0.0113 | -0.0234 | +0.0016 | -0.0142 | × |
| MCC | -0.0657 | -0.0457 | -0.0375 | -0.0517 | -0.0206 | -0.0442 | × |
| DICE | -0.0044 | -0.0093 | -0.0011 | -0.0117 | +0.0043 | -0.0045 | × |
| VOLSIM | -0.0064 | -0.0114 | -0.0031 | -0.0143 | +0.0032 | -0.0064 | × |
| HD95 | +3.9398 | +4.0085 | +1.6498 | +4.8103 | -0.6837 | +2.7449 | × |

→ 개선 0/7 · 평균ΔMCC -0.0442 · **미충족**
(참고: 구 eval ÷52 MCC 평균Δ -0.0114)

## val · 시드 [0, 1, 2, 3, 4]

| 지표 | s0 | s1 | s2 | s3 | s4 | 평균Δ | 개선 |
|---|---|---|---|---|---|---|---|
| PRECISION | -0.0251 | -0.0018 | -0.0219 | -0.0426 | +0.0050 | -0.0173 | × |
| RECALL | -0.0455 | -0.0455 | -0.0455 | -0.0455 | -0.0455 | -0.0455 | × |
| F1 | -0.0455 | -0.0455 | -0.0434 | -0.0556 | -0.0434 | -0.0467 | × |
| MCC | -0.0345 | -0.0079 | -0.0322 | -0.0433 | -0.0023 | -0.0240 | × |
| DICE | -0.0368 | -0.0302 | -0.0362 | -0.0440 | -0.0294 | -0.0353 | × |
| VOLSIM | -0.0368 | -0.0300 | -0.0362 | -0.0474 | -0.0293 | -0.0359 | × |
| HD95 | +14.5500 | +14.5132 | +13.8214 | +18.9300 | +13.7847 | +15.1198 | × |

→ 개선 0/7 · 평균ΔMCC -0.0240 · **미충족**
(참고: 구 eval ÷52 MCC 평균Δ -0.0282)

**판정 (test:미충족·val:미충족) → 미채택**

## 4. 판정

- 1차 병변 TP Δ -20 → 미충족
- 안전 test 수정 50/파손 57 · val 수정 40/파손 48 → 위반
- 2차 ΔMCC test -0.0442 · val -0.0240 → 미충족

**→ 기각**

⚠ 룰은 ICA 원위 병변만 건드린다. τ3.0 은 train 격자 {0.5,1,2,3} 의 끝값이다(격자는 결과 전 고정).
