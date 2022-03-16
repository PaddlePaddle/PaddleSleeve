# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weighted ensemble attack on yolov3 model using cw attack"""

from __future__ import absolute_import

import os
import sys

import paddle
from obj_detection.attack.utils.tools import get_model, denormalize_image, bcolors, plot_image_objectdetection_ppdet
from ppdet.data.source.dataset import ImageFolder
from ppdet.data.reader import TestReader

import numpy as np
import copy

work_dir = os.path.abspath(os.getcwd())
sys.path.append(work_dir)


def parse_summary():
    """Parse dictionary from summary json file."""
    import json
    with open(
            os.path.dirname(os.path.realpath(__file__)) +
            '/../utils/summary.json') as f:
        summary = json.load(f)

    summary['models'] = list(set(summary['paddledet_models']))
    return summary


def to_tanh_space(x, mid, range):
    """Convert an input from model space to tanh space."""
    # map from [min_, max_] to [-1, +1]
    x = (x - mid) / range

    # from [-1, +1] to approx. (-1, +1)
    x = x * 0.999999

    # from (-1, +1) to (-inf, +inf)
    return paddle.atan(x)


def run_attack(models, weight_list, data, target_class, eps=10 / 255):
    """
    Use several models' logits to generate a adversary example for cw attack.
    Args:
        models (list) : list of models used to train an adversarial example
        weights (list) : weight of each model
        data (dict) : input data
        target_class (int) : target class to be attacked
        eps (float) : maximum perturbation
    Returns:
        adv_img (Tensor) : adversarial example
    """
    adv_img = denormalize_image(data['image'])

    # attack configs
    min_, max_ = (0, 1)
    mid_point = (min_ + max_) * 0.5
    half_range = (max_ - min_) * 0.5
    C, H, W = (3, 608, 608)
    norm = 'l2'
    c_init = 0.05
    c_lower_bound = 0
    c_upper_bound = 10
    c_search_steps = 10
    max_iterations = 10
    learning_rate = 0.05
    abort_early = True
    total_weight = sum(weight_list)

    # attack start
    tanh_original = to_tanh_space(adv_img, mid_point, half_range)

    # the binary search finds the smallest const for which we
    # find an adversarial
    const = c_init

    # will be close but not identical to a.original_image
    reconstructed_original = paddle.tanh(tanh_original) * half_range + mid_point

    best_lp = None
    best_perturbed = None
    best_pred = None

    # Outer loop for linearly searching for c
    for c_step in range(c_search_steps):
        if c_step == c_search_steps - 1 and c_search_steps >= 10:
            const = c_upper_bound

        tanh_pert = copy.deepcopy(tanh_original)
        tanh_pert.stop_gradient = False

        small_lp = None
        small_perturbed = None
        small_pred = None

        # create a new optimizer to minimize the perturbation
        optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=[tanh_pert])

        for iteration in range(max_iterations):
            is_adv = False
            optimizer.clear_grad()
            x = paddle.tanh(tanh_pert) * half_range + mid_point
            adv_loss = 0
            for i, m in enumerate(models):
                m._data = data
                x_norm = paddle.unsqueeze(m._preprocessing(paddle.squeeze(x)), axis=0)
                features = m._gather_feats(x_norm)
                model_adv_loss = m.adv_loss(features=features, target_class=target_class)
                model_adv_loss = paddle.mean(paddle.clip(model_adv_loss, min=0))
                adv_loss += model_adv_loss * weight_list[i]

            adv_loss /= total_weight
            if not adv_loss > 0:
                is_adv = True

            if norm == 'l2':
                lp_loss = paddle.norm(x - reconstructed_original, p=2) / np.sqrt(C * H * W)
            else:
                lp_loss = paddle.max(paddle.abs(x - reconstructed_original)).astype(np.float64)

            if is_adv:
                if small_lp is None or lp_loss < small_lp:
                    small_lp = lp_loss
                    small_perturbed = x
                    small_pred = features['bbox_pred'].numpy()
                if abort_early and small_lp < eps:
                    return small_perturbed

            loss = lp_loss + const * adv_loss
            loss.backward()
            optimizer.step()

        if small_perturbed is not None:
            c_upper_bound = const
            if best_lp is None or small_lp < best_lp:
                best_lp = small_lp
                best_perturbed = small_perturbed
                best_pred = small_pred
        else:
            c_lower_bound = const

        const_new = (c_lower_bound + c_upper_bound) / 2.0
        const = const_new

    return best_perturbed


if __name__ == '__main__':
    summary = parse_summary()

    # choose from available models:
    # "paddledet_yolov3_darknet53",
    # "paddledet_yolov3_mobilenet_v1",
    # "paddledet_yolov3_mobilenet_v3_large",
    # "paddledet_yolov3_resnet50vd",
    # "paddledet_faster_rcnn_resnet50",
    # "paddledet_faster_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnet101_vd_fpn",
    # "paddledet_cascade_rcnn_resnet50_fpn",
    # "paddledet_faster_rcnn_resnext101_64x4d_dcn",
    # "paddledet_detr_resnet50",
    # "paddledet_deformable_detr_resnet50"
    model_list = ['paddledet_yolov3_darknet53',
                  'paddledet_detr_resnet50',
                  'paddledet_faster_rcnn_resnet50_fpn']
    weight_list = [1, 1, 1]

    victim_model_name = 'paddledet_yolov3_mobilenet_v3_large'
    dataset_dir = os.path.dirname(os.path.realpath(__file__ + '../../')) + '/utils/images/ensemble_demo'

    victim_model = None
    # victim_model = get_model(victim_model_name, 'paddledet', summary)

    models = []
    for m in model_list:
        model = get_model(m, 'paddledet', summary)
        model.eval()
        models.append(model)
    # victim model
    if victim_model is None:
        victim_model = get_model(victim_model_name, 'paddledet', summary)

    test_dataset = ImageFolder()
    test_dataset.set_images([dataset_dir])
    test_loader = TestReader(sample_transforms=[{'Decode': {}},
                                                {'Resize': {'target_size': [608, 608], 'keep_ratio': False,
                                                            'interp': 2}},
                                                {'NormalizeImage': {'mean': [0.485, 0.456, 0.406],
                                                                    'std': [0.229, 0.224, 0.225], 'is_scale': True}},
                                                {'Permute': {}}],
                             batch_transforms=[{'PadMaskBatch': {'pad_to_stride': -1, 'return_pad_mask': True}}])(
        test_dataset, 0)

    num_test = len(test_loader)
    eps = 15 / 255
    attack_success = 0
    orig_conf = []
    transfer_conf = []
    best_data = None
    best_adv = None
    best_target = None
    max_diff = 0

    print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
    for i, data in enumerate(test_loader):
        if i > num_test:
            break

        ori_img = copy.deepcopy(data['image'])
        # find target class
        probe_model = models[0]
        outs = probe_model._model(data)
        bbox = outs['bbox'].numpy()
        max_idx = np.argmax(bbox[:, 1])
        if bbox[max_idx, 1] < 0.1:
            continue
        target_class = int(bbox[max_idx, 0])

        adv_img = run_attack(models, weight_list, data, target_class, eps)
        if adv_img is None:
            print('Attack failed')
            continue

        attack_success += 1
        print("Attack succeed")
        data['image'] = ori_img
        orig_bbox = victim_model._model(data)['bbox'].numpy()
        tgt_idx = np.argwhere(orig_bbox[:, 0] == target_class)[:, 0]
        max_conf = np.max(orig_bbox[tgt_idx][:, 1]) if tgt_idx.size > 0 else 0
        orig_conf.append(max_conf)

        # Testing transferability
        data['image'] = paddle.unsqueeze(victim_model._preprocessing(paddle.squeeze(adv_img)), axis=0)
        adv_bbox = victim_model._model(data)['bbox'].numpy()
        adv_tgt_idx = np.argwhere(adv_bbox[:, 0] == target_class)[:, 0]
        adv_max_conf = np.max(adv_bbox[adv_tgt_idx][:, 1]) if adv_tgt_idx.size > 0 else 0
        transfer_conf.append(adv_max_conf)

        if max_conf - adv_max_conf > max_diff:
            max_diff = max_conf - adv_max_conf
            print("New Best Transfer example, {}".format(data['im_id'].numpy()))
            best_data = data
            best_target = int(target_class)
            best_adv = {'adv_img': paddle.squeeze(adv_img).numpy(),
                        'ori_img': paddle.squeeze(denormalize_image(ori_img)).numpy(),
                        'adv_bbox': adv_bbox,
                        'ori_bbox': orig_bbox}

    print(bcolors.HEADER + bcolors.UNDERLINE + 'Attack Summary:' + bcolors.ENDC)
    print("{0} images attacked \n"
          "Average confidence of victim model on original image: {1}\n"
          "Average confidence of victim model on transferred adversarial image: {2}"
          .format(attack_success, np.mean(np.array(orig_conf)), np.mean(np.array(transfer_conf))))

    if best_adv is not None and best_data is not None and best_target is not None:
        img_dir = os.path.dirname(os.path.realpath(__file__ + '../..')) + '/outputs/images'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        victim_model._data = best_data
        plot_image_objectdetection_ppdet(kmodel=victim_model,
                                         adv_dict=best_adv,
                                         title=best_target,
                                         figname="{0}/best_weighted_cw_transfer_result_im_id_{1}"
                                         .format(img_dir, best_data['im_id'].numpy()))
        print('Visualization result is saved in %s' % img_dir)

    print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)
    print('\n')
