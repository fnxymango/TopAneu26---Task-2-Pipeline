"""250-epoch variant of the fork's SkelRecall + NoMirroring trainer (fast fold-0 vessel check).
NOTE: nnU-Net introspects self.__init__ signature via locals() to record my_init_kwargs,
so the __init__ MUST use explicit params (no *args/**kwargs), matching the parent."""
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerSkeletonRecallNoMirroring)


class nnUNetTrainerSkeletonRecallNoMirroring_250ep(nnUNetTrainerSkeletonRecallNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
