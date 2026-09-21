"""1000-epoch(nnU-Net 표준 기본값) variant — D800이 250ep로 이미 Dice 0.703을 낸 뒤,
전체 학습budget을 다 쓰면 더 오르는지 보는 단계. D7xx의 crop->1000ep 패턴과 동일한 이유.
NOTE: nnU-Net introspects self.__init__ signature via locals() to record my_init_kwargs,
so the __init__ MUST use explicit params (no *args/**kwargs), matching the parent."""
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerSkeletonRecallNoMirroring)


class nnUNetTrainerSkeletonRecallNoMirroring_1000ep(nnUNetTrainerSkeletonRecallNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
