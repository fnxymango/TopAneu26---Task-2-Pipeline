import os
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_shape = properties_dict['shape_after_cropping_and_before_resampling']
    new_spacing = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]

    # ---- TopAneu 저메모리 경로 (2026-08-29) ---------------------------------
    # 37클래스 로짓을 한꺼번에 리샘플하면 resample_data_or_seg 내부의
    # data.astype(float) 이 전 채널을 float64 로 승격시켜 10 GB 단위로 튄다
    # (GC 컨테이너 메모리 한도 초과의 직접 원인).
    # 채널을 하나씩 리샘플하며 argmax 를 누적하면 결과가 **완전히 동일**하다:
    #   · resample_data_or_seg 는 이미 채널별 독립 루프다
    #   · softmax 는 단조라 argmax(softmax(x)) == argmax(x)
    # region 기반 라벨(has_regions)이나 확률 반환 요청 시에는 원래 경로를 쓴다.
    _lowmem = (not return_probabilities) and (not label_manager.has_regions)

    # GPU 리샘플 경로: 큰 케이스(최대 1억 3천만 복셀)에서 CPU 경로는 RAM 10 GB / 케이스당 289초로
    # 컨테이너 한도(8 GB · 7분)를 둘 다 넘긴다. 채널별로 GPU 삼선형 보간을 쓰면 둘 다 해결된다.
    # 단 skimage resize 와 비트 단위로 같지는 않다 → val 전수 검증 후 채택할 것.
    # do_separate_z 가 필요한 이방성 데이터에서는 의미가 달라지므로 CPU 경로로 되돌린다.
    _use_gpu = False
    if _lowmem and os.environ.get("TOPANEU_GPU_RESAMPLE", "1") == "1" and torch.cuda.is_available():
        # force_separate_z 는 plans 의 configuration 딕셔너리에서 직접 읽는다.
        # (ConfigurationManager 에 *_kwargs 속성은 없다 — 예전 코드가 여기서 조용히 실패했다)
        try:
            from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis
            _kw = {}
            try:
                _kw = configuration_manager.configuration.get('resampling_fn_probabilities_kwargs', {}) or {}
            except Exception:
                _kw = {}
            _sepz, _ = determine_do_sep_z_and_axis(_kw.get('force_separate_z', None),
                                                   current_spacing, new_spacing)
            _use_gpu = not _sepz
        except Exception as _e:
            print(f'    [GPU-RESAMPLE] 사용 불가 → CPU 경로: {type(_e).__name__}: {_e}', flush=True)
            _use_gpu = False
    if _lowmem:
        print(f'    [GPU-RESAMPLE] {"ON" if _use_gpu else "OFF"}', flush=True)

    if _lowmem:
        import numpy as _np
        _C = predicted_logits.shape[0]
        _best = None
        _idx = None
        _gbest = None
        _gidx = None
        _tshape = tuple(int(v) for v in target_shape)
        for _c in range(_C):
            if _use_gpu:
                # 누적까지 GPU 에서 한다. 채널마다 CPU 로 내리면 호스트 RAM 이 1 GB 넘게 더 든다
                # (최대 케이스 기준). 최종 라벨맵(uint8) 만 한 번 내려받는다.
                _src = predicted_logits[_c:_c + 1]
                if not isinstance(_src, torch.Tensor):
                    _src = torch.from_numpy(_np.ascontiguousarray(_src))
                _g = _src.to("cuda", dtype=torch.float32, non_blocking=True).unsqueeze(0)
                _o = torch.nn.functional.interpolate(_g, size=_tshape, mode="trilinear",
                                                     align_corners=False)[0, 0]
                del _g, _src
                if _gbest is None:
                    _gbest = _o.clone()
                    _gidx = torch.zeros(_tshape, dtype=torch.uint8, device=_o.device)
                else:
                    _gm = _o > _gbest
                    _gbest[_gm] = _o[_gm]
                    _gidx[_gm] = _c
                    del _gm
                del _o
                continue
            else:
                _src = predicted_logits[_c:_c + 1]
                if isinstance(_src, torch.Tensor):
                    _src = _src.cpu().numpy()
                _r = configuration_manager.resampling_fn_probabilities(
                    _src, target_shape, current_spacing, new_spacing)
                if isinstance(_r, torch.Tensor):
                    _r = _r.cpu().numpy()
                _r = _r[0]
            if _best is None:
                _best = _r.astype(_np.float32, copy=True)
                _idx = _np.zeros(_best.shape, dtype=_np.uint8 if _C < 256 else _np.uint16)
            else:
                _m = _r > _best
                _np.copyto(_best, _r, where=_m)
                _np.copyto(_idx, _np.uint8(_c) if _C < 256 else _np.uint16(_c), where=_m)
                del _m
            del _r
        if _use_gpu:
            _idx = _gidx.cpu().numpy()
            del _gbest, _gidx
            torch.cuda.empty_cache()
        del predicted_logits, _best
        segmentation = _idx
    else:
        predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                                target_shape, current_spacing, new_spacing)
        # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
        # apply_inference_nonlin will convert to torch
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        del predicted_logits
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    # # needed for cascade
    # if isinstance(predicted, str):
    #     assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
    #                               "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(predicted)
    #     predicted = np.load(predicted)
    #     os.remove(del_file)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
    torch.set_num_threads(old_threads)
