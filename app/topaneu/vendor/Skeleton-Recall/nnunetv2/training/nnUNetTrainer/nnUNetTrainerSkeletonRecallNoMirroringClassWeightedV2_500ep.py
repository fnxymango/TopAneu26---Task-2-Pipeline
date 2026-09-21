"""V2 — class weight 2차 조정 (2026-08-12).

V1(`...ClassWeighted_500ep`, 22=5x / 15,16,31=3x) 결과:
  Dice 0.7485 -> 0.7793 (nnU-Net 지표), 0점 클래스 4개 -> 2개.
  L-P3P4(22) 0 -> 0.7375, R-AChA(31) 0 -> 0.5057 로 구제 성공.
  3rd-A2(15) / 3rd-A3(16)만 여전히 0.

2026-08-12 확률맵 진단(val 4케이스, GT 영역 softmax 확인):
  15/16의 확률 질량이 사실상 0(전 voxel 4위 이하, 1등이 되려면 ~1e8배 필요).
  argmax 승자가 배경이 아니라 **좌/우 대응 클래스**로 일관됨:
    3rd-A3 -> L-A3 66~69% / R-A3 26~27%
    3rd-A2 -> L-A1A2, R-A1A2 (각 43%)
  => "덜 배워서 확률이 낮다"가 아니라 좌/우 클래스와 구분 근거가 약한
     클래스 정의 모호성 문제로 보인다. 따라서 가중치 상향의 기대값은 낮지만,
     레버를 하나만 바꾸는 확인 실험으로 15/16만 3x -> 10x 올린다.

V1에서 이미 구제된 22(5x)/31(3x)은 건드리지 않는다 — 잘 되는 것을 흔들지 않기 위함.

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
    22: 5.0,    # L-P3P4  — V1에서 구제 성공(0 -> 0.7375). 유지
    31: 3.0,    # R-AChA  — V1에서 구제 성공(0 -> 0.5057). 유지
    15: 10.0,   # 3rd-A2  — V1 3x로 실패. 이번 실험의 유일한 변경점
    16: 10.0,   # 3rd-A3  — 상동
}


class nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep(nnUNetTrainerSkeletonRecallNoMirroring_500ep):
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
