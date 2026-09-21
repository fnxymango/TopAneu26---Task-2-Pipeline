"""Tversky+TopK-CE base 레시피를 500epoch로. 1000ep는 STEP9/10(5fold 앙상블)까지
합치면 너무 길어서(약 4일), 500ep로 먼저 fold0을 보고 250ep 대비 개선폭이 이미
수렴 근처면 그대로 5fold 앙상블까지 500ep로 진행 — 1000ep는 스킵.
num_epochs를 늘리면 nnU-Net의 PolyLR이 그 값 기준으로 새로 감쇠 스케줄을 잡으므로
250ep를 도중에 끊는 것과 다르게 500ep 자체로 완결된 학습이다.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTverskyTopkCE import nnUNetTrainerTverskyTopkCE


class nnUNetTrainerTverskyTopkCE_500ep(nnUNetTrainerTverskyTopkCE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 500
