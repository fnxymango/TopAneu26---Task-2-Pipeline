# C36 — ExtraTrees 위치 분류기 (52클래스)

**2026-08-18 기준 C 계열 최고.** 지표·모델 선택 기준은 MCC(PROJECT_RULES.md §6-1).

## 0. 계보

| | |
|---|---|
| baseline | `C2_location_weighted_knn` (dist_mm 34차원 kNN) |
| prev | `C31_conf_gated_beta` (RandomForest + 확신 게이트) |
| **직전 대비 바뀐 레버** | **RandomForest → ExtraTrees. 그 외 전부 동일** |
| 목적 | C21 의 분류기 비교(GBM/앙상블 기각)가 β 이중보정 상태의 train CV 판정이었으므로, 교정된 설정에서 val e2e 로 8종 재시험 |

## 1. 성능 — test 83

분모 두 가지를 모두 적는다. `covered_gt` = `adjusted_div_present`(GT 등장 36클래스로 나눔, 기록 기준),
`official_div52` = 조직위 evaluate.py 그대로(52로 나눔, 리더보드 비교용).

| 분모 | Precision | Recall | **MCC** | Dice | VolSim | HD95↓ | 복합(aligned) |
|---|---|---|---|---|---|---|---|
| **covered_gt ÷36** | 0.3551 | 0.4090 | **0.3609** | 0.2205 | 0.2408 | 0.7033 | 0.3139 |
| official ÷52 | 0.2458 | 0.2832 | **0.2499** | 0.1527 | 0.1667 | 0.4869 | 0.2686 |

val 42 (÷33): MCC **0.4791** covered / 0.3040 official.

## 2. 직전(RF) 대비

| test | off.MCC | cov.MCC | cov.Dice | cov.HD95↓ | cov.복합 |
|---|---|---|---|---|---|
| C31 RandomForest | 0.2289 | 0.3306 | 0.2029 | 0.7221 | 0.2887 |
| **C36 ExtraTrees** | **0.2499** | **0.3609** | **0.2205** | **0.7033** | **0.3139** |
| 차이 | +9.2% | **+9.2%** | +8.7% | −0.0188 | +8.7% |

**6지표 전부 개선.** val 에서도 et 0.3040 > rf 0.2940 이었으므로 선택 절차도 정상이다.

## 3. 분류기 8종 비교 (val 42, official MCC)

| 모델 | val MCC |
|---|---|
| **ExtraTrees** | **0.3040** |
| RandomForest (직전) | 0.2940 |
| RF+ET 확률평균 | 0.2935 |
| RF+GB / HistGB | 0.2831 |
| RF+ET+GB | 0.2823 |
| MLP (128, alpha 1e-2) | 0.1902 |
| LogisticRegression | 0.1232 |

읽을 점 셋:

1. **선형·신경망이 처참하다.** 피처가 선형 분리되지 않고, 268샘플 × 43클래스에 MLP 는
   클래스당 6샘플로 파라미터를 맞추는 셈이라 성립하지 않는다.
2. **GBM 은 RF 이하** — C21 의 기각은 결론 자체는 옳았다.
3. **앙상블이 손해다.** rf_et(0.2935) < et 단독(0.3040). 약한 RF 를 평균에 섞어 ET 를 끌어내린다.

ET 가 이긴 이유는 데이터 조건 자체다 — 268샘플 × 112차원에서는 최적 분할을 찾아
표본 노이즈에 맞추는 것(RF)보다 분할점을 무작위로 뽑아 분산을 낮추는 쪽(ET)이 유리하다.

## 4. 같이 확인하고 기각한 것

| 실험 | 내용 | 결과 |
|---|---|---|
| C38 | 확신 게이트 τ 유무 | **노이즈.** τ=0.5 0.3609 / τ=0 0.3597 / τ=0.6 0.3561. val 에서 본 +0.014 는 42케이스 잡음 |
| C39 | ET 하이퍼파라미터 (max_features × min_samples_leaf) | **기각.** val 최고 mf=0.7(0.3100)이 test 에서 0.1755 로 붕괴. 기본 sqrt 유지 |
| C39 | 추론 시 미러 TTA | **기각.** val 0.2876 < 0.3040 |

<!-- C39 는 val 42케이스에서 0.006 차이를 근거로 고른 결과가 test 에서 −0.074 로 뒤집힌
     사례다. 이 split 에서 0.03 미만의 val 차이는 선택 근거가 되지 못한다. -->

## 5. 남은 헤드룸이 어디인가

| | cov.MCC | top-1 |
|---|---|---|
| 분류 천장 (GT 병변 입력, ET) | **0.5167** | 0.663 |
| 현재 e2e | 0.3609 | — |
| **검출이 먹는 몫** | **0.1558** | |

검출기 실측(프로덕션 `probavgf`): test 병변 민감도 **0.733 (63/86)** · FP 24 (0.29/case) ·
복셀 Dice 0.636 · 음성 22케이스에서 FP 5.

**86개 중 23개를 아예 못 찾는다.** MCC 는 1복셀만 겹쳐도 TP 라 Dice 는 거의 기여하지 않고
저 23개가 그대로 손실이다. 그리고 후처리로는 못 줄인다 — C27 에서 확률임계 5종을 재봤으나
민감도가 33/43 으로 전부 동일했고(확률맵 자체 천장이 val 35/43), C28 에서 필터를 풀어
0.814 까지 올렸더니 FP 가 11→38 로 늘어 MCC 가 오히려 떨어졌다.

**다음 레버는 검출기 재학습이다.** A7(Tversky α 0.3→0.15, β 0.7→0.85) 진행 중.

## 6. 재현

```bash
export TOPANEU_ROOT=<project>
cd $TOPANEU_ROOT/code/sblee/nnunet/scripts
python c5_location_v2.py eval \
  --train-feat ../analysis/c10_feat_train.json --split test \
  --vessel-dir  $TOPANEU_ROOT/experiments/_c1_realpred/vespp_test \
  --bp-dir      $TOPANEU_ROOT/experiments/_c4_bpgraph/vespp_test \
  --aneurysm-pred-dir $TOPANEU_ROOT/experiments/_c1_realpred/aneu_test_probavgf \
  --model et --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 --tag c36_et
```
