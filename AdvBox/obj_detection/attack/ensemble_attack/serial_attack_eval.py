#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
attacking successing rate on paddle model
"""
import copy
import os.path
import sys

sys.path.append("../..")
import paddle
import numpy as np
from ppdet.data.reader import TestReader
from ppdet.data.source.dataset import ImageFolder
from attack.utils.tools import get_model, get_distance, get_metric, get_criteria
from attack.utils.tools import denormalize_image, plot_image_objectdetection_ppdet, bcolors


def parse_summary():
    """Parse dictionary from summary json file."""
    import json
    import os
    with open(
            os.path.dirname(os.path.realpath(__file__)) +
            '/../utils/summary.json') as f:
        summary = json.load(f)

    summary['models'] = list(set(summary['paddledet_models']))
    return summary


def run_attack(method, model, image, dist='l2', eps=10 / 255, target_cls=0):
    distance = get_distance(dist)
    criteria = get_criteria('target_class_miss', target_cls, model_name='paddledet_')
    attack = get_metric(method, model, criteria, distance)
    attack._default_threshold = eps
    model.eval()

    # attacking start
    if method == 'pgd':
        adversary = attack(image, label, unpack=False, epsilons=eps)
    else:
        adversary = attack(image, label, unpack=False, abort_early=True)

    if adversary.image is not None and adversary.distance.value < eps:
        return adversary.image
    else:
        return None


if __name__ == "__main__":
    summary = parse_summary()

    # Choose your model here
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

    model_list = ["paddledet_yolov3_darknet53", "paddledet_faster_rcnn_resnet50_fpn"]
    victim_model_name = "paddledet_yolov3_mobilenet_v3_large"

    dataset_dir = os.path.dirname(os.path.realpath(__file__ + '/..')) + '/utils/images/ensemble_demo'

    # Change the attacking config below
    dist = 'l2'
    attack_method = 'carlini_wagner'
    eps = 10 / 255

    # You need not change anything below
    models = []
    for model_name in model_list:
        model = get_model(model_name, 'paddledet', summary)
        models.append(model)
    vic = get_model(victim_model_name, 'paddledet', summary)
    vic.eval()

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
    attack_success = 0
    orig_conf = []
    transfer_conf = []
    best_data = None
    best_adv = None
    max_diff = 0

    print(bcolors.BOLD + 'Process start' + bcolors.ENDC)

    for i, data in enumerate(test_loader):
        if i > num_test:
            break

        ori_img = copy.deepcopy(data['image'])

        probe_model = models[0]
        probe_model._data = data
        outs = probe_model._model(data)
        bbox = outs['bbox']
        max_idx = np.argmax(bbox[:, 1])
        if bbox[max_idx, 1] < 0.1:
            continue
        label = int(bbox[max_idx, 0])

        pert_img = denormalize_image(data['image']).numpy()

        for model in models:
            model._data = data
            adv_img = run_attack(method=attack_method,
                                 model=model,
                                 image=pert_img,
                                 dist=dist,
                                 eps=eps,
                                 target_cls=label)
            if adv_img is not None:
                pert_img = adv_img

        print("Attack Finished")

        # Testing transferability
        pert_img_tensor = vic._preprocessing(paddle.to_tensor(pert_img, dtype='float32'))
        data['image'] = paddle.unsqueeze(pert_img_tensor, axis=0)
        adv_bbox = vic._model(data)['bbox'].numpy()
        adv_tgt_idx = np.argwhere(adv_bbox[:, 0] == label)[:, 0]
        adv_max_conf = np.max(adv_bbox[adv_tgt_idx][:, 1]) if adv_tgt_idx.size > 0 else 0
        transfer_conf.append(adv_max_conf)

        data['image'] = ori_img
        orig_bbox = vic._model(data)['bbox'].numpy()
        tgt_idx = np.argwhere(orig_bbox[:, 0] == label)[:, 0]
        max_conf = np.max(orig_bbox[tgt_idx][:, 1]) if tgt_idx.size > 0 else 0
        orig_conf.append(max_conf)

        if (max_conf - adv_max_conf > max_diff):
            max_diff = max_conf - adv_max_conf
            print("New Best Transfer example, {}".format(data['im_id'].numpy()))
            best_data = data
            best_adv = {'adv_img': pert_img,
                        'ori_img': denormalize_image(ori_img).numpy(),
                        'adv_bbox': adv_bbox,
                        'ori_bbox': orig_bbox}

    print('\n')
    print(bcolors.HEADER + bcolors.UNDERLINE + 'Attack Summary:' + bcolors.ENDC)

    print("{0} images attacked \n"
          "Average confidence of victim model on original image: {1}\n"
          "Average confidence of victim model on transferred adversarial image: {2}"
          .format(num_test, np.mean(np.array(orig_conf)), np.mean(np.array(transfer_conf))))

    if best_data is not None and best_adv is not None:
        img_dir = os.path.dirname(os.path.realpath(__file__ + '/..')) + '/outputs/images'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        vic._data = best_data
        plot_image_objectdetection_ppdet(kmodel=vic, adv_dict=best_adv,
                                         figname="{0}/best_transfer_result_im_id_{1}".format(img_dir, best_data['im_id'].numpy()))
        print('Visualization result is saved in %s' % img_dir)

    print(bcolors.BLUE + 'Process Finished' + bcolors.ENDC)
