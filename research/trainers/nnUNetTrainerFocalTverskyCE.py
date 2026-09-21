"""
Focal-Tversky + CE trainer for TopAneu aneurysm (tiny-lesion detection).

E2 loss lever — ONE change vs the D520 base (nnUNetTrainerTverskyCE):
    base D520 : Tversky(0.3/0.7)        + CE
    this      : Focal-Tversky(0.3/0.7, gamma=1.33) + CE

The region term is modulated by (1 - TverskyIndex)^gamma so the gradient
focuses on hard / under-segmented lesions (gamma=1 recovers plain Tversky;
gamma>1 focuses harder). FN-weighting (beta>alpha) from the base is kept.

Everything else — plain-z dataset (520), oversample 0.60, 250 epochs, plans,
architecture, augmentation — is inherited unchanged from nnUNetTrainerTverskyCE,
so this is a clean one-lever comparison. Run on Dataset520_TopAneuBinary,
3d_fullres, fold 0.

Source of truth lives in sblee/nnunet/trainers/; symlinked into the nnunetv2
package so nnU-Net's trainer discovery can import it.
"""
import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTverskyCE import nnUNetTrainerTverskyCE
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientFocalTverskyLoss(nn.Module):
    """Focal-Tversky loss: FTL = (1 - TverskyIndex)^gamma.

    TverskyIndex = TP / (TP + alpha*FP + beta*FN); beta>alpha penalizes FN.
    gamma>1 focuses the loss on hard (low-index) examples; gamma=1 == Tversky.
    Mirrors MemoryEfficientTverskyLoss but returns the focal-modulated value.
    """

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 ddp=True, alpha=0.3, beta=0.7, gamma=1.33):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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
        focal = (1 - tversky).clamp_min(0) ** self.gamma
        return focal.mean()


class nnUNetTrainerFocalTverskyCE(nnUNetTrainerTverskyCE):
    ALPHA = 0.3
    BETA = 0.7
    GAMMA = 1.33

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerFocalTverskyCE assumes argmax (non-region) labels"
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False,
             'ddp': self.is_ddp, 'alpha': self.ALPHA, 'beta': self.BETA, 'gamma': self.GAMMA},
            {}, weight_ce=1, weight_dice=1, ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientFocalTverskyLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
