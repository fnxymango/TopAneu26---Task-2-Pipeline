"""500-epoch variant — 1000ep는 5fold 앙상블까지 합치면 약 5일이라 너무 길어서,
500ep로 fold0을 먼저 보고 250ep(Dice 0.7027) 대비 개선폭이 이미 수렴 근처면
그대로 5fold 앙상블까지 500ep로 진행 — 1000ep는 스킵.
NOTE: nnU-Net introspects self.__init__ signature via locals() to record my_init_kwargs,
so the __init__ MUST use explicit params (no *args/**kwargs), matching the parent."""
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerSkeletonRecallNoMirroring)


class nnUNetTrainerSkeletonRecallNoMirroring_500ep(nnUNetTrainerSkeletonRecallNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
