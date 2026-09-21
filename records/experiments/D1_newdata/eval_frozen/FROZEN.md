# 동결된 공식 eval

출처: https://github.com/Bangulli/TopAneu-26.git
커밋: **660da7ab699446b58748dec4bf6fef6e5efd34ab** (660da7a)
동결 시각: 2026-09-11 10:35:22 KST

## 왜 동결하나
2026-09-11 10:27 KST, H6 채점이 도는 **도중에** `~/TopAneu-26/eval/task2/evaluate.py` 가
갱신됐다(F1 추가). 이미 떠 있던 프로세스는 옛 모듈을 메모리에 들고 있어 F1 없이 기록했고,
이후 배치는 F1 을 기록한다 — **한 실험 안에서 지표 집합이 갈렸다.**
채점에 쓰는 eval 은 반드시 고정본이어야 하고, 결과에 커밋 해시를 남겨야 한다.

## 60765a5 → 660da7a 변경 (evaluate.py)
F1 추가**뿐**이다. 기존 6지표 계산식은 한 글자도 안 바뀌었다.
```
+ aggregates[f"F1_{i}"] = 2*tp/((2*tp)+fp+fn) if (tp+fp+fn) else np.nan
- for metric in ["PRECISION","RECALL","MCC","DICE","HD95","VOLSIM"]:
+ for metric in ["PRECISION","RECALL","F1","MCC","DICE","HD95","VOLSIM"]:
```
따라서 **기존 6지표 비교는 전부 유효하다.** F1 만 새로 채우면 된다.

## 지표 구조 (개정판)
분모가 0 이면 NaN 이고 `nanmean` 이 분모에서 제외한다. 그래서 지표마다 유효 클래스 수가 다르다.
GT 에 없는 클래스를 예측했을 때:
  PRECISION = 0/(0+fp) = 0      → **유효값 0 이 평균에 들어간다 (손해)**
  F1        = 0/(0+fp+0) = 0    → **유효값 0 이 들어간다 (손해)**
  RECALL    = tp+fn=0 → NaN     → 제외
  MCC       = 분모 0 → NaN      → 제외
즉 **F1 은 MCC 가 못 잡는 환각을 잡는다.** 판정에 F1 을 넣어야 하는 실질적 이유다.
