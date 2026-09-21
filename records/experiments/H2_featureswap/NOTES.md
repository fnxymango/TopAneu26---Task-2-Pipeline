# H2 — 검출기 freeze · 분류기 피처만 개정판으로 (2026-09-10, 진행 중)

## 왜

E9 는 검출기·피처·gC **세 가지가 동시에** 바뀌어 있어 기여도를 귀속할 수 없다.
검출기를 제출본 것으로 고정하면 (a) 런타임을 전혀 안 건드리고 얻는 몫이 얼마인지 나오고,
(b) ResEncL 의 순수 기여도가 분리된다.

```
b1on    구검출기 + 구피처   + gC ON    ← 제출본 (H1 에서 측정 완료)
  │ ① 피처 효과        ← 채택 판정 대상 · 런타임 비용 0
b1Non   구검출기 + 개정피처 + gC ON
  │ ② gC 효과
b1Noff  구검출기 + 개정피처 + gC OFF
  │ ③ 검출기 효과      ← "ResEncL 차이가 큰가" 의 답 · 런타임 1.5~1.7배
e9off   E9검출기 + 개정피처 + gC OFF   ← H1 에서 측정 완료
```

## 무엇을 바꿨나 — 딱 하나

`c5_location_v2.py eval --train-feat` 를 `e11_feat_hyb_ov.json` → **`e11_feat_hyb_ov_NEW.json`** 로.
검출 마스크는 양쪽 다 `aneu_{sp}_b1ff` 로 **동일**하다. 혈관·분기점·c7·필터도 전부 동일.

피처 json 은 **GT 병변에서 뽑은 학습셋**이라 검출기와 무관하다. 그래서 구 검출기에 개정판
피처로 학습한 RF 를 그대로 물릴 수 있다. 두 json 의 차이는 학습 데이터 개정(2026-09-01 판)뿐이다.

gC 파라미터는 `TOPANEU_TOPK` 만 2↔1 로 바꾸고 나머지(vox 3 · ica 1 · margin 0.7)는 고정.

## 판정규칙 — 결과 보기 전에 `H2.sh` 머리말에 고정

채택 판정은 ① 에만 적용: `b1Non_pf − b1on_pf` 가 신 eval 6지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 을
test·val 둘 다 만족하면 채택. 제출본이 이미 필터를 갖고 있으므로 `_pf` 끼리 비교하는 것이 실제 조건이다.
②·③ 은 분해용이며 채택 판정이 아니다. 시드 산포가 평균보다 크면 결론에 명시한다.

## 반영 방법 (① 이 이득으로 나올 경우)

컨테이너 안의 **파일 두 개 교체**로 끝난다. 검출기도 모델 tar 도 그대로다.
```
/opt/app/topaneu/code/sblee/nnunet/analysis/final_rf_seed3.pkl      → 개정피처로 학습한 RF
/opt/app/topaneu/code/sblee/nnunet/analysis/e11_feat_hyb_ov.json    → e11_feat_hyb_ov_NEW.json
```
`pipeline_case.py` 는 pkl 안의 `topk` 를 읽어 gC 를 켜므로, ② 결과에 따라 pkl 을 만들 때
`topk.n` 을 1 로 넣을지 2 로 둘지 정한다.

## 결과

(진행 중 — `report/RESULTS_H2.md`)
