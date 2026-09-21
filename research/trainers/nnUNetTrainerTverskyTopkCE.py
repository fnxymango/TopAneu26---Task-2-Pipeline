"""
Tversky + TopK-CE trainer for TopAneu aneurysm (tiny-lesion detection).

E2 loss lever — ONE change vs the D520 base (nnUNetTrainerTverskyCE):
    base D520 : Tversky(0.3/0.7) + CE      (CE over ALL voxels)
    this      : Tversky(0.3/0.7) + TopK-CE (CE over hardest k=10% voxels)

Only the CE companion is swapped: TopK-CE keeps the loss on the hardest 10%
of voxels (lesion + boundary + confusing background) instead of being swamped
by the easy background mass of a tiny-lesion task. The region (Tversky) term is
identical to the base -> clean one-lever comparison.

Everything else — plain-z dataset (520), oversample 0.60, 250 epochs, plans,
architecture, augmentation — is inherited unchanged from nnUNetTrainerTverskyCE.
Run on Dataset520_TopAneuBinary, 3d_fullres, fold 0.

Source of truth lives in sblee/nnunet/trainers/; symlinked into the nnunetv2
package so nnU-Net's trainer discovery can import it.
"""
import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTverskyCE import (
    nnUNetTrainerTverskyCE, MemoryEfficientTverskyLoss)
from nnunetv2.training.loss.robust_ce_loss import TopKLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1


class Tversky_and_topk_loss(nn.Module):
    """Tversky (region) + TopK cross-entropy (hardest k% voxels).

    Mirrors nnU-Net's DC_and_topk_loss but uses MemoryEfficientTverskyLoss as the
    region term instead of SoftDiceLoss, so alpha/beta FN-weighting is preserved.
    """

    def __init__(self, tversky_kwargs, topk_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            topk_kwargs['ignore_index'] = ignore_label
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ce = TopKLoss(**topk_kwargs)
        self.dc = MemoryEfficientTverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, \
                'ignore label is not implemented for one hot encoded target variables (Tversky_and_topk_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class nnUNetTrainerTverskyTopkCE(nnUNetTrainerTverskyCE):
    ALPHA = 0.3
    BETA = 0.7
    K = 10  # percent of hardest voxels kept by TopK-CE

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerTverskyTopkCE assumes argmax (non-region) labels"
        loss = Tversky_and_topk_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False,
             'ddp': self.is_ddp, 'alpha': self.ALPHA, 'beta': self.BETA},
            {'k': self.K, 'label_smoothing': 0.0},
            weight_ce=1, weight_dice=1, ignore_label=self.label_manager.ignore_label)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
