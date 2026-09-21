"""
Custom nnU-Net v2 trainer for tiny-lesion detection (TopAneu aneurysm).

Differences vs base nnUNetTrainer:
  - Loss: Tversky(alpha=0.3, beta=0.7) + CE  (base: SoftDice + CE)
          beta>alpha penalizes false-negatives more -> pushes the model to
          predict foreground instead of collapsing to all-background.
  - oversample_foreground_percent: 0.60  (base 0.33)
  - num_epochs: 250  (base 1000; faster feedback for the detection check)

Everything else (plans, architecture, normalization, augmentation) unchanged.
Intended for non-region (argmax) datasets, e.g. Dataset520_TopAneuBinary.
Source of truth lives in sblee/nnunet/trainers/; symlinked into the nnunetv2
package so nnU-Net's trainer discovery can import it.
"""
import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientTverskyLoss(nn.Module):
    """Tversky loss (Dice generalization). Tversky = TP / (TP + alpha*FP + beta*FN).
    beta > alpha => false negatives cost more. Mirrors MemoryEfficientSoftDiceLoss."""

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 ddp=True, alpha=0.3, beta=0.7):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            tp = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)
        fp = sum_pred - tp
        fn = sum_gt - tp

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth).clamp_min(1e-8)
        return -tversky.mean()


class nnUNetTrainerTverskyCE(nnUNetTrainer):
    ALPHA = 0.3
    BETA = 0.7

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.60
        self.num_epochs = 250

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerTverskyCE assumes argmax (non-region) labels"
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False,
             'ddp': self.is_ddp, 'alpha': self.ALPHA, 'beta': self.BETA},
            {}, weight_ce=1, weight_dice=1, ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientTverskyLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
