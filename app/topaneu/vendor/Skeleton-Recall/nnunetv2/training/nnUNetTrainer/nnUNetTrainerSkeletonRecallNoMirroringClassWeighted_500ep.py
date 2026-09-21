"""D800 fold0(500ep) 사후분석(2026-08-11): L-P3P4가 epoch~40-104엔 pseudo dice 0.5~0.76로
정상 학습되다가 epoch105에서 한 배치 만에 0.0으로 붕괴, 이후 395epoch 내내 회복 못함
(같은 epoch에 잠깐 흔들린 R-M3/L-M3는 곧 회복함 — L-P3P4만 dead class 상태로 고착).
데이터는 충분(fold0 train 292케이스 중 280개=96%에 존재, 평균 1783voxel/case로 R-P3P4보다도 큼)
하므로 데이터 부족이 아니라 학습 붕괴 문제 -> CE loss에 class weight를 줘서 예측이 0으로
꺼진 뒤에도 복귀 그래디언트가 죽지 않게 함(Dice loss만으론 pred=0일 때 그래디언트가 거의 0).
3rd-A2/3rd-A3/R-AChA는 500epoch 내내 pseudo dice가 한 번도 0.05를 넘은 적 없는 순수 데이터
부족 클래스라 가중치 효과는 제한적이겠지만, 비용이 들지 않아 같이 살짝 올려둠.

NOTE: nnU-Net introspects self.__init__ signature via locals() to record my_init_kwargs,
so the __init__ MUST use explicit params (no *args/**kwargs), matching the parent.
"""
import warnings

import torch

from nnunetv2.training.loss.compound_losses import DC_SkelREC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSkeletonRecallNoMirroring_500ep import (
    nnUNetTrainerSkeletonRecallNoMirroring_500ep)

import numpy as np

# dataset.json label id 기준 (Dataset800_TopAneuVessel417)
CLASS_WEIGHT_OVERRIDES = {
    22: 5.0,   # L-P3P4 — 학습 중 붕괴(dead class), 데이터는 충분해서 가중치로 복구 시도
    15: 3.0,   # 3rd-A2 — 순수 데이터부족, 참고용으로 소폭만
    16: 3.0,   # 3rd-A3 — 상동
    31: 3.0,   # R-AChA — 상동
}


class nnUNetTrainerSkeletonRecallNoMirroringClassWeighted_500ep(nnUNetTrainerSkeletonRecallNoMirroring_500ep):
    def _build_loss(self):
        if self.label_manager.ignore_label is not None:
            warnings.warn('Support for ignore label with Skeleton Recall is experimental and may not work as expected')

        n_classes = self.label_manager.num_segmentation_heads
        weight = np.ones(n_classes, dtype=np.float32)
        for cls_id, w in CLASS_WEIGHT_OVERRIDES.items():
            if cls_id < n_classes:
                weight[cls_id] = w
        weight_t = torch.from_numpy(weight).to(self.device)

        loss = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            soft_skelrec_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={'weight': weight_t}, weight_ce=1, weight_dice=1, weight_srec=self.weight_srec,
            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
