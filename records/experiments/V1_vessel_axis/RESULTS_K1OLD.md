# K1OLD — 구 학습표 위 K1 e2e (시드 0~4) · 규칙은 K1OLD.sh 머리말 고정

# K0 판정 — `b1on_pf` → `b1k1o_pf` (시드 0,1,2,3,4 · test+val 130단위)

장치 영가설 보정: 거짓 양성률 0.000 → 사용 가능

- 단위별 적중 순증: 오른 단위 **7** · 내린 단위 **2** · 단측 부호검정 p = **0.0898**
- 5시드 합: ΔTP +8 · ΔFP(오답 클래스) +8
  · test: 오른 4 · 내린 2
  · val: 오른 3 · 내린 0

| split | 케이스 | 클래스 | 적중 시드 순증 |
|---|---|---|---|
| val | center5_mr_023 | 1.5 VA-BA junction | +4 |
| test | center1_mr_877 | R-1.3 VA-PICA junction | +1 |
| test | center2_ct_184 | L-5.3 M1-M2 junction | +1 |
| test | center2_ct_192 | R-3.5 ICA C7-AChA-junction | +1 |
| test | center2_mr_086 | R-3.3 ICA C6-nonOA | -1 |
| test | center4_ct_067 | 4.1 Acom complex | -1 |
| test | center5_mr_007 | 1.5 VA-BA junction | +1 |
| val | center2_ct_169 | R-5.1 M1 trunk | +1 |
| val | center4_ct_134 | L-3.5 ICA C7-AChA-junction | +1 |
- 참고(판정 미사용): 기준에서 오답 0 이던 클래스에 새로 생긴 오답 20 (새로 맞힌 ΔTP +8 대비)

- 주판정(p<0.05 ∧ n+>n−): 미충족 · 안전(ΔFP ≤ ΔTP): 충족

**K0 판정 → 미채택**

## 참고 · 7지표 (판정 미사용 · 기준 b1on_pf 대비 평균 Δ · HD95 는 낮을수록 개선)

| split | PRECISION | RECALL | F1 | MCC | DICE | VOLSIM | HD95 | 개선 |
|---|---|---|---|---|---|---|---|---|
| test | -0.0437 | +0.0123 | +0.0019 | -0.0280 | +0.0082 | +0.0087 | -0.5860 | 5/7 |
| val | +0.0364 | +0.0364 | +0.0228 | +0.0473 | -0.0011 | +0.0019 | -4.6032 | 6/7 |
