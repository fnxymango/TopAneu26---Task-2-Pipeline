"""Tversky+TopK-CE base 레시피를 250ep -> 1000ep(nnU-Net 표준 기본값)로 되돌린 변형.

nnUNetTrainerTverskyCE가 "빠른 피드백"을 위해 250ep로 낮춰뒀던 것(원 주석 참고)을,
crop/레시피가 검증된 뒤 전체 학습budget을 다 쓰는 단계에서 사용.
로스/아키텍처/증강 등 다른 모든 것은 nnUNetTrainerTverskyTopkCE와 동일 — 이 한 줄만 다르다.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTverskyTopkCE import nnUNetTrainerTverskyTopkCE


class nnUNetTrainerTverskyTopkCE_1000ep(nnUNetTrainerTverskyTopkCE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 1000
