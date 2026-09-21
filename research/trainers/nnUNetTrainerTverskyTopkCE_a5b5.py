"""
FP-rebalanced variant of nnUNetTrainerTverskyTopkCE for the 417-case migration.

D720 base recipe (alpha=0.3 FP-weight, beta=0.7 FN-weight) was tuned on the old
98-case dataset where every case had >=1 real aneurysm -> tolerating FP to chase
recall was free. The 417-case release adds 111 aneurysm-free cases (27%, all
center1 MRA); on D720 val42, 6/11 negative cases got a hallucinated FP blob
(Dice=0 each), which is a direct consequence of alpha<beta telling the loss
"a false alarm costs less than a miss" even when there is nothing to miss.

ONE lever vs D720: alpha 0.3->0.5, beta 0.7->0.5 (balanced FP/FN weighting).
Everything else (TopK-CE k=10, oversample 0.6, plans, augmentation) unchanged.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTverskyTopkCE import nnUNetTrainerTverskyTopkCE


class nnUNetTrainerTverskyTopkCE_a5b5(nnUNetTrainerTverskyTopkCE):
    ALPHA = 0.5
    BETA = 0.5
