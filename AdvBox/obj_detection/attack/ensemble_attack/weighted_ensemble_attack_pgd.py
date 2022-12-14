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

"""Weighted ensemble attack on yolov3 model using pgd attack"""

from __future__ import absolute_import

import copy
import os
import sys
work_dir = os.path.abspath(os.getcwd())
sys.path.append(work_dir)

import paddle
from attack.utils.tools import get_model, denormalize_image, bcolors, plot_image_objectdetection_ppdet
from ppdet.data.source.dataset import ImageFolder
from ppdet.data.reader import TestReader
import numpy as np
import copy


def parse_summary():
    """Parse dictionary from summary json file."""
    import json
    with open(
            os.path.dirname(os.path.realpath(__file__)) +
            '/../utils/summary.json') as f:
        summary = json.load(f)

    summary['models'] = list(set(summary['paddledet_models']))
    return summary


def run_attack(models, weight_list, data, target_class, eps=10/255, eps_step=4/255, steps=20):
    """
    Use several models' logits to generate a adversary example for cw attack.
    Args:
        models (list) : list of models used to train an adversarial example
        weights (lsit) : weight of each model
        data (dict) : input data
        target_class (int) : target class to be attacked
        eps (float) : maximum perturbation
    Returns:
        adv_img (Tensor) : adversarial example
    """
    _min, _max = (0, 1)
    C, H, W = (3, 608, 608)
    norm = 'L2'
    ori_img = denormalize_image(data['image'])
    adv_img = ori_img
    for (k, v) in data.items():
        v.stop_gradient = True

    for step in range(steps):
        grad = paddle.zeros_like(ori_img)
        # loss_total = 0
        for i, m in enumerate(models):
            adv_img.stop_gradient = False
            adv_img.clear_gradient()
            m._data = data
            x_norm = paddle.unsqueeze(m._preprocessing(paddle.squeeze(adv_img)), axis=0)
            features = m._gather_feats(x_norm)
            loss = m.adv_loss(features=features, target_class=target_class)

            if paddle.all(loss <= 0):
                adv_img.clear_gradient()
                adv_img.stop_gradient = True
                continue
            loss.backward()
            gradient = - adv_img.grad * weight_list[i]
            adv_img.stop_gradient = True
            grad = paddle.add(grad, gradient)

        grad.stop_gradient = True
        if paddle.all(grad == 0):
            return adv_img
        if norm == 'L2':
            count_pix = np.sqrt(C * H * W * (_max - _min) ** 2)
            l2_norm = paddle.norm(grad, p=2)
            normalized_grad = grad / l2_norm * count_pix
            adv_img = adv_img + normalized_grad * eps_step
            eta = adv_img - ori_img
            mse_eta = paddle.norm(eta, p=2) / count_pix
            if mse_eta > eps:
                eta *= eps / mse_eta
            adv_img = paddle.clip(ori_img + eta, _min, _max)
            # print(eta)
            # print(adv_img_tensor)
        elif norm == 'Linf':
            normalized_grad = paddle.sign(grad)
            adv_img = adv_img + normalized_grad * eps_step
            eta = paddle.clip(adv_img - ori_img, -eps, eps)
            adv_img = paddle.clip(ori_img + eta, _min, _max)

    return None

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
    weight_list = [3, 1, 1]

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

    # Attack configs
    num_test = len(test_loader)
    eps = 15 / 255
    eps_step = 4 / 255
    steps = 20
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

        adv_img = run_attack(models, weight_list, data, target_class, eps, eps_step, steps)
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

    print('\n')
    print(bcolors.HEADER + bcolors.UNDERLINE + 'Attack Summary:' + bcolors.ENDC)
    print("{0} images attacked \n"
          "Average confidence of victim model on original image: {1}\n"
          "Average confidence of victim model on transferred adversarial image: {2}"
          .format(attack_success, np.mean(np.array(orig_conf)), np.mean(np.array(transfer_conf))))
    print('\n')

    if best_adv is not None and best_data is not None and best_target is not None:
        img_dir = os.path.dirname(os.path.realpath(__file__ + '../..')) + '/outputs/images'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        victim_model._data = best_data
        plot_image_objectdetection_ppdet(kmodel=victim_model,
                                         adv_dict=best_adv,
                                         title=best_target,
                                         figname="{0}/best_weighted_pgd_transfer_result_im_id_{1}"
                                         .format(img_dir, best_data['im_id'].numpy()))
        print('Visualization result is saved in %s' % img_dir)

    print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)
    print('\n')

