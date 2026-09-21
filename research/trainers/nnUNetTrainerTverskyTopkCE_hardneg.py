"""
Hard-negative (domain-aware) case oversampling for the 417-case migration.

D720 base recipe samples training CASES uniformly (nnU-Net's default
sampling_probabilities=None in get_dataloaders -> uniform np.random.choice over
tr_keys). The 417-case release's 111 aneurysm-free cases are ALL from one new
center (center1, MRA) -- so "aaneurysm-free" is confounded with "unseen scanner
domain". On D720 val42, 6/11 aneurysm-free cases got a hallucinated FP blob
(Dice=0 each): the hypothesis is the model hasn't seen enough of the center1
distribution (positive or negative) per epoch under uniform case sampling,
since center1 is ~48% of train cases by count but was never specifically
emphasized.

ONE lever vs D720 base (nnUNetTrainerTverskyTopkCE, alpha=0.3/beta=0.7 unchanged):
override get_dataloaders() to set per-case sampling_probabilities so center1
cases are HARD_NEG_WEIGHT times more likely to be drawn than everyone else.
With HARD_NEG_WEIGHT=3.0 and 140/292 train cases from center1, effective
per-epoch exposure to center1 patches becomes ~73% (vs ~48% uniform).

This is orthogonal to the Tversky alpha/beta rebalance (nnUNetTrainerTverskyTopkCE_a5b5)
tried in parallel -- keep them as separate single-lever experiments, not stacked,
so results stay interpretable.

Everything else (loss, oversample_foreground_percent for patch-centering, plans,
augmentation, epochs) is inherited unchanged from nnUNetTrainerTverskyTopkCE.
"""
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTverskyTopkCE import nnUNetTrainerTverskyTopkCE
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter


class nnUNetTrainerTverskyTopkCE_hardneg(nnUNetTrainerTverskyTopkCE):
    HARD_NEG_TAG = 'center1'   # substring matched against case identifiers, e.g. topaneu_center1_mr_050
    HARD_NEG_WEIGHT = 3.0      # relative per-case sampling weight for tagged cases vs everyone else

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # --- hard-negative / hard-domain oversampling (the one lever) ---
        tr_keys = dataset_tr.identifiers
        weights = np.array([self.HARD_NEG_WEIGHT if self.HARD_NEG_TAG in k else 1.0 for k in tr_keys])
        sampling_probabilities = (weights / weights.sum()).astype(np.float64)
        n_tagged = int((weights == self.HARD_NEG_WEIGHT).sum())
        self.print_to_log_file(
            f"hardneg oversampling: {n_tagged}/{len(tr_keys)} train cases tagged '{self.HARD_NEG_TAG}', "
            f"weight x{self.HARD_NEG_WEIGHT} -> effective per-epoch exposure to tagged cases = "
            f"{sampling_probabilities[weights == self.HARD_NEG_WEIGHT].sum():.3f} "
            f"(vs {n_tagged / len(tr_keys):.3f} under uniform sampling)")

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=sampling_probabilities, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
