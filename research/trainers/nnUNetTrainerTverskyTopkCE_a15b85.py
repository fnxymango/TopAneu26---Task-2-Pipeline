"""재현율을 더 밀어붙인 Tversky 변형 — A7 (2026-08-17).

근거(오늘 측정): test 에서 분류 천장이 covered MCC 0.4429 인데 e2e 가 0.3233 이다.
차이 0.120 이 **검출이 놓친 병변** 값이고, 이건 지금 파이프라인 단일 최대 손실이다.
그리고 C27/C28 에서 확률임계 5종 x 필터 3종을 다 재본 결과 후처리로는 못 줄인다 —
확률맵 자체의 민감도 천장이 val 35/43(0.814) 이고 현행이 33/43 이다.
즉 손실을 줄이려면 **검출기를 다시 학습**해야 한다.

ONE lever vs nnUNetTrainerTverskyTopkCE: alpha 0.3->0.15, beta 0.7->0.85.
Tversky 에서 alpha 는 FP 가중, beta 는 FN 가중이므로 놓침의 비용을 더 키운다.
TopK-CE k=10, oversample 0.6, plans, augmentation 등 나머지는 전부 그대로.

주의 — 반대 방향 선례가 있다. nnUNetTrainerTverskyTopkCE_a5b5 는 417코호트에
동맥류 없는 케이스가 27%(111케이스) 들어오면서 FP 환각이 문제가 돼 alpha 를 0.5 로
**올린** 변형이다. 그러니 이 실험은 FP 가 늘어날 것이 확실하고, 질문은
"늘어난 FP 를 c7 혈관거리 필터가 걸러낸 뒤에도 민감도 순증이 남는가" 다.
그래서 평가는 Dice 가 아니라 **c7 필터 적용 후 병변단위 민감도 / FP** 로 한다.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTverskyTopkCE import nnUNetTrainerTverskyTopkCE


class nnUNetTrainerTverskyTopkCE_a15b85(nnUNetTrainerTverskyTopkCE):
    ALPHA = 0.15
    BETA = 0.85
