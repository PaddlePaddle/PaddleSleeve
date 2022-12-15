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

"""Launcher tools."""

from attack.utils.image import load_image, draw_bounding_box_on_image
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFile, ImageDraw
from ppdet.utils.colormap import colormap
from ppdet.data.source.category import get_categories
import numpy as np

class bcolors:
    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    CYAN = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD = "\033[;1m"
    REVERSE = "\033[;7m"
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_image_format(framework_name, model_name):
    """Return the correct input range and shape for target framework and model"""
    special_shape = {'paddledet': {'paddledet_yolov3_darknet53': (608, 608)}}
    special_bound = {}
    default_shape = (224, 224)
    default_bound = (0, 1)
    if special_shape.get(framework_name, None):
        if special_shape[framework_name].get(model_name, None):
            default_shape = special_shape[framework_name][model_name]
    if special_bound.get(framework_name, None):
        if special_bound[framework_name].get(model_name, None):
            default_bound = special_bound[framework_name][model_name]
    return {'shape': default_shape, 'bounds': default_bound}


def get_model(model_name, framework, summary=None):
    """Get model dispatcher."""
    switcher = {'paddledet': lambda: _load_paddledet_model(model_name, summary)
                # May add more frameworks/models here
    }
    _get_model = switcher.get(framework, None)
    return _get_model()


def get_distance(distance_name):
    """Get the distance metric."""
    import attack.utils.distances as distances
    switcher = {
        'mse': distances.MSE,
        'mae': distances.MAE,
        'linf': distances.Linf,
        "l0": distances.L0,
        "l2": distances.MSE
    }
    return switcher.get(distance_name, None)


def get_metric(attack_name, model, criteria, distance):
    """Get the attack class object."""
    import attack.single_attack as metrics
    kwargs = {
        'model': model,
        'criterion': criteria,
        'distance': distance,
    }
    switcher = {
        "carlini_wagner": lambda x: metrics.CarliniWagnerMetric(**x),
        "pgd": lambda x: metrics.ProjectedGradientDescentMetric(**x)
    }
    _init_attack = switcher.get(attack_name, None)
    attack = _init_attack(kwargs)
    return attack


def get_criteria(criteria_name, target_class=None, prob=None, model_name=None):
    """Get the adversarial criteria."""
    import attack.utils.criteria as criteria
    switcher = {
        "target_class_miss": lambda: criteria.TargetClassMiss(target_class, model_name)
    }
    return switcher.get(criteria_name, None)()


def _load_paddledet_model(model_name, summary=None):
    import attack.models.paddledet as ppdet
    switcher = {
        'paddledet_yolov3_darknet53': 'yolov3_darknet53_270e_coco',
        'paddledet_yolov3_mobilenet_v3_large': 'yolov3_mobilenet_v3_large_270e_coco',
        'paddledet_yolov3_resnet50vd': 'yolov3_r50vd_dcn_270e_coco',
        'paddledet_yolov3_mobilenet_v1': 'yolov3_mobilenet_v1_270e_coco',
        'paddledet_faster_rcnn_resnet50': 'faster_rcnn_r50_1x_coco',
        'paddledet_faster_rcnn_resnet50_fpn': 'faster_rcnn_r50_fpn_1x_coco',
        'paddledet_faster_rcnn_resnet101_vd_fpn': 'faster_rcnn_r101_vd_fpn_1x_coco',
        'paddledet_faster_rcnn_resnext101_64x4d_dcn': 'faster_rcnn_dcn_x101_vd_64x4d_fpn_1x_coco',
        'paddledet_cascade_rcnn_resnet50_fpn': 'cascade_rcnn_r50_fpn_1x_coco',
        'paddledet_detr_resnet50': 'detr_r50_1x_coco',
        'paddledet_deformable_detr_resnet50': 'deformable_detr_r50_1x_coco'
    }

    cfg_name = switcher.get(model_name, None)
    if cfg_name.startswith('yolov3'):
        model = ppdet.PPdet_Yolov3_Model(cfg_name)
    elif cfg_name.startswith('faster'):
        model = ppdet.PPdet_Rcnn_Model(cfg_name)
    elif cfg_name.startswith('cascade'):
        model = ppdet.PPdet_Rcnn_Model(cfg_name, cascade=True)
    elif cfg_name.__contains__('detr'):
        model = ppdet.PPdet_Detr_Model(cfg_name)
    return model


def plot_image_objectdetection(adversary, kmodel, bounds=(0, 1), title=None, figname='compare.png'):
    """Plot the images."""
    from attack.utils.image import draw_letterbox
    pred_ori = kmodel.predictions(adversary.original_image)
    pred_adv = kmodel.predictions(adversary.image)
    class_names = kmodel.get_class()

    ori_image = draw_letterbox(adversary.original_image, pred_ori, class_names=class_names, bounds=bounds)

    adv_image = draw_letterbox(adversary.image, pred_adv, class_names=class_names, bounds=bounds)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.axis('off')

    ax1.imshow(ori_image)
    ax1.set_title('Origin')
    ax1.axis('off')

    ax2.imshow(adv_image)
    ax2.set_title('Adversary')
    ax2.axis('off')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.9)

    # in case you do not have GUI interface
    plt.savefig(figname, bbox_inches='tight', dpi=1000)

    plt.show()


def plot_image_objectdetection_ppdet(adv=None,
                                     kmodel=None,
                                     adv_dict=None,
                                     threshold=0.5,
                                     title=None,
                                     figname='compare.png',
                                     mask=-1):
    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    def _draw_bbox(image, bboxes, threshold):
        """
        Draw bbox on image
        """
        draw = ImageDraw.Draw(image)

        catid2color = {}

        color_list = colormap(rgb=True)[:40]
        for dt in np.array(bboxes):
            catid, bbox, score = int(dt[0]), dt[2:], dt[1]
            if catid == mask or score < threshold:
                continue

            if catid not in catid2color:
                idx = np.random.randint(len(color_list))
                catid2color[catid] = color_list[idx]
            color = tuple(catid2color[catid])

            # draw bbox
            xmin, ymin, xmax, ymax = bbox
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill=color)
            # draw label
            catid = clsid2catid[catid+1]
            text = "{} {:.2f}".format(catid2name[catid], score)
            tw, th = draw.textsize(text)
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

        return image

    if adv is None:
        if adv_dict is None :
            return
        prev = adv_dict['ori_img']
        after = adv_dict['adv_img']
        prev_bbox = adv_dict['ori_bbox']
        after_bbox = adv_dict['adv_bbox']
    else:
        if adv.image is None:
            return
        prev = adv.original_image
        after = adv.image
        prev_bbox = kmodel.predictions(prev)
        after_bbox = adv._best_adversarial_output

    anno_file = kmodel._dataset.get_anno()
    _, catid2name = get_categories(
        'coco', anno_file=anno_file)
    diff = np.absolute(prev - after)
    if diff.max() == 0:
        scale = 1
    else:
        scale = 1 / diff.max()
    diff = diff * scale

    adv_image = Image.fromarray((np.transpose(after, [1, 2, 0]) * 255).astype('uint8'))
    ori_image = Image.fromarray((np.transpose(prev, [1, 2, 0]) * 255).astype('uint8'))
    diff_image = Image.fromarray((np.transpose(diff, [1, 2, 0]) * 255).astype('uint8'))
    scale_factor = np.squeeze(kmodel._data['scale_factor'].numpy())
    ih, iw = scale_factor
    w, h = ori_image.size
    ori_shape = (int(w / iw), int(h / ih))
    # print("h {}, w {}".format(h, w), ori_shape)
    adv_image = adv_image.resize(ori_shape, Image.ANTIALIAS)
    ori_image = ori_image.resize(ori_shape, Image.ANTIALIAS)
    diff_image = diff_image.resize(ori_shape, Image.ANTIALIAS)

    adv_image = _draw_bbox(adv_image, after_bbox, threshold=0.5)
    ori_image = _draw_bbox(ori_image, prev_bbox, threshold=0.3)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.axis('off')

    ax1.imshow(ori_image)
    ax1.set_title('Origin')
    ax1.axis('off')

    ax2.imshow(adv_image)
    ax2.set_title('Adversary')
    ax2.axis('off')

    ax3.imshow(diff_image)
    ax3.set_title('Diff * %.1f' % scale)
    ax3.axis('off')

    if title is not None:
        if isinstance(title, int):
            title = "Attack target: {}".format(catid2name[clsid2catid[title+1]])
            print(title)
        fig.suptitle(title, fontsize=10, fontweight='bold', y=0.80)
    plt.savefig(figname, bbox_inches='tight', dpi=600)


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """

    Args:
        image : 3D or 4D Tensor
        mean:
        std:

    Returns:
        3D Tensor

    """
    import paddle
    if len(image.shape) > 3:
        image = paddle.squeeze(image)
    for i in range(image.shape[0]):
        image[i] = image[i] * std[i] + mean[i]
    image = paddle.clip(image, min=0, max=1)
    return image
