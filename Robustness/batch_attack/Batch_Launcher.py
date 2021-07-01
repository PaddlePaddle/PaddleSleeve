# coding=UTF-8
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
from __future__ import absolute_import

import os
import sys
import argparse
import numpy as np
import pandas as pd
import random

work_dir = os.path.abspath(os.getcwd())
sys.path.append(work_dir)

from perceptron.utils.tools import get_image
from perceptron.utils.tools import get_metric
from perceptron.utils.tools import get_distance
from perceptron.utils.tools import get_criteria
from perceptron.utils.tools import get_model
from perceptron.launcher import validate_cmd, parse_summary
from perceptron.utils.tools import bcolors

import urllib3
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

here = os.path.dirname(os.path.abspath(__file__))


def read_line(path):
    """Read file auxiliary function."""
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    data = {}
    for i in lines:
        t = i.split(r' ', 1)
        data[t[0].strip()] = t[1].strip()
    return data


def read_line2(path):
    """Read file auxiliary function."""
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    data = {}
    for idx, i in enumerate(lines):
        data[idx] = i.strip()
    return data


def read_images(args, data_dir, rel2id):
    """All images in one folder: get imagenet dev dataset images and labels."""
    if not os.path.exists(data_dir):
        print(bcolors.RED + 'Please provide a valid data dir: ' + bcolors.ENDC, data_dir)
        exit(-1)
    channel_axis = 'channels_first' if (args.framework == 'pytorch' or args.framework == 'paddle') else 'channels_last'

    # List all image names in the folder
    data_file_names = os.listdir(data_dir)

    # Need to change 1: Read the corresponding relationship between the image name and its label
    annotations = os.path.join(here, 'caffe_ilsvrc12/val.txt')
    synset = os.path.join(here, 'caffe_ilsvrc12/synsets.txt')
    words = os.path.join(here, 'caffe_ilsvrc12/synset_words.txt')
    file_name2id = read_line(annotations)  # 'ILSVRC2012_val_00049992.JPEG': '357'
    id2class = read_line2(synset)  # 970: 'n09193705'
    class2class_name = read_line(words)  # 'n07730033': 'cardoon'

    dataset = []
    label = []
    data_names = []
    data_num = len(data_file_names)
    if data_num < 100:
        choice = range(data_num)
    else:
        choice = random.sample(range(data_num), 100)  # randomly select 100 samples in the folder
    for idx in choice:
        fn = data_file_names[idx]
        image_path = os.path.join(data_dir, fn)
        image = get_image(image_path, args.framework, args.model, channel_axis)
        dataset.append(image)
        # Need to change 2: Convert the image name to the corresponding label
        label.append(rel2id[class2class_name[id2class[int(file_name2id[fn])]]])
        data_names.append(fn)
    return dataset, label, data_names


def run_attack(args, summary, data_dir=os.path.join(here, 'ILSVRC2012_img_val')):
    """Attack the samples sampled in the dev dataset one by one, and write the results to csv."""
    model = get_model(args.model, args.framework, summary)
    distance = get_distance(args.distance)
    criteria = get_criteria(args.criteria, args.target_class)
    metric = get_metric(args.metric, model, criteria, distance)
    keywords = [args.framework, args.model, args.criteria, args.metric, args.distance]

    # Load original data
    with open(os.path.join(here, '../perceptron/utils/labels.txt')) as info:
        id2rel = eval(info.read())
        rel2id = {v: k for k, v in id2rel.items()}

    # # Read image, label and corresponding pic_name
    images, true_labels, pic_names = read_images(args, data_dir, rel2id)
    print('dataset length: ', len(images))

    res_list = []
    for idx, image in tqdm(enumerate(images)):
        res = {}
        ori_label = np.argmax(model.predictions(image))
        res['pic_name'] = pic_names[idx]
        res['label'] = id2rel[true_labels[idx]]
        res['ori_pred_label'] = id2rel[ori_label]
        if args.model in ["yolo_v3", "keras_ssd300", "retina_resnet_50"]:
            adversary = metric(image, unpack=False, binary_search_steps=1)
        elif args.metric not in ["carlini_wagner_l2", "carlini_wagner_linf"]:
            if args.metric in summary['verifiable_metrics']:
                adversary = metric(image, ori_label, unpack=False, verify=args.verify)
            else:
                adversary = metric(image, ori_label, unpack=False, epsilons=1000)
        else:
            adversary = metric(image, ori_label, unpack=False, binary_search_steps=10, max_iterations=5)
        if adversary.image is None:
            res['adv_pred_label'] = 'Cannot find an adversary!'
            res['distance_' + args.distance] = ''
        else:
            res['adv_pred_label'] = id2rel[np.argmax(model.predictions(adversary.image))]
            res['distance_' + args.distance] = str(adversary.distance).split()[-1]

        if args.metric in ["brightness", "rotation", "horizontal_translation",
                           "vertical_translation"]:
            res['verifiable_bound'] = str(adversary.verifiable_bounds)

        res_list.append(res)

    if args.metric in ["brightness", "rotation", "horizontal_translation",
                       "vertical_translation"]:
        df = pd.DataFrame(res_list,
                          columns=['pic_name', 'label', 'ori_pred_label', 'adv_pred_label', 'distance_' + args.distance,
                                   'verifiable_bound'])
    else:
        df = pd.DataFrame(res_list,
                          columns=['pic_name', 'label', 'ori_pred_label', 'adv_pred_label', 'distance_' + args.distance])

    df.to_csv('{}.csv'.format('_'.join(keywords)), index=False)


if __name__ == "__main__":
    summary = parse_summary()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help="specify the name of the model you want to evaluate",
        default="resnet50"
    )
    parser.add_argument(
        "-e", "--metric",
        help="specify the name of the metric",
        default="gaussian_blur"
    )
    parser.add_argument(
        "-d", "--distance",
        help="specify the distance metric to evaluate adversarial examples",
        choices=summary['distances'],
        default="mse"
    )
    parser.add_argument(
        "-f", "--framework",
        help="specify the deep learning framework used by the target model",
        choices=summary['frameworks'],
        default="paddle"
    )
    parser.add_argument(
        "-c", "--criteria",
        help="specify the adversarial criteria for the evaluation",
        choices=summary['criterions'],
        default="misclassification"
    )
    parser.add_argument(
        "--verify",
        help="specify if you want a verifiable robustness bound",
        action="store_true"
    )
    parser.add_argument(
        "-t", "--max_iterations",
        help="specify the maximum number of iterations for the attack",
        type=int,
        default="1000"
    )
    parser.add_argument(
        "--target_class",
        help="Used for detection/target-attack tasks, indicating the class you want to vanish",
        type=int
    )

    args = parser.parse_args()
    validate_cmd(args)
    run_attack(args, summary)
