# R3 — 접합 거리 상한(τ4mm) e2e (시드 0~4) · 규칙은 R3.sh 머리말 고정

# K0 판정 — `b1Non_pf` → `b1r3_pf` (시드 0,1,2,3,4 · test+val 130단위)

장치 영가설 보정: 거짓 양성률 0.000 → 사용 가능

- 단위별 적중 순증: 오른 단위 **3** · 내린 단위 **6** · 단측 부호검정 p = **0.9102**
- 5시드 합: ΔTP -18 · ΔFP(오답 클래스) -19
  · test: 오른 2 · 내린 2
  · val: 오른 1 · 내린 4

| split | 케이스 | 클래스 | 적중 시드 순증 |
|---|---|---|---|
| test | center1_mr_028 | R-3.6 ICA C7-nonBranch | +5 |
| test | center4_ct_063 | L-3.4 ICA C7-Pcom-junction | -5 |
| test | center5_mr_466 | L-3.2 ICA C6-OA-junction | -5 |
| val | center2_ct_166 | R-3.2 ICA C6-OA-junction | -5 |
| val | center2_mr_083 | L-3.2 ICA C6-OA-junction | -5 |
| val | center4_ct_134 | L-3.4 ICA C7-Pcom-junction | -5 |
| val | center5_mr_023 | 1.5 VA-BA junction | +4 |
| val | center2_ct_169 | R-5.1 M1 trunk | -3 |
| test | center4_ct_067 | 4.1 Acom complex | +1 |
- 참고(판정 미사용): 기준에서 오답 0 이던 클래스에 새로 생긴 오답 8 (새로 맞힌 ΔTP -18 대비)

- 주판정(p<0.05 ∧ n+>n−): 미충족 · 안전(ΔFP ≤ ΔTP): 충족

**K0 판정 → 미채택**

## 참고 · 7지표 (판정 미사용 · 기준 b1Non_pf 대비 평균 Δ · HD95 는 낮을수록 개선)

| split | PRECISION | RECALL | F1 | MCC | DICE | VOLSIM | HD95 | 개선 |
|---|---|---|---|---|---|---|---|---|
| test | +0.0468 | -0.0018 | +0.0120 | +0.0169 | +0.0049 | +0.0067 | -3.3619 | 6/7 |
| val | -0.0116 | -0.0697 | -0.0705 | -0.0095 | -0.0399 | -0.0450 | +22.8796 | 0/7 |
