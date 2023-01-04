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
HopSkipJumpAttack tutorial on cifar10.
"""
from __future__ import print_function
import os
import sys
import cv2
sys.path.append("../..")

import logging
logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import argparse
import numpy as np
import functools
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize
from paddle.vision.transforms import ToTensor
from past.utils import old_div
from adversary import Adversary
from examples.utils import add_arguments, print_arguments, show_images_diff
from examples.utils import bcolors
from attacks.hop_skip_jump_attack import HopSkipJumpAttack
from models.blackbox import PaddleBlackBoxModel
from models.whitebox import PaddleWhiteBoxModel

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('target_image', str, None, 'The type of successful attack, e.g., input/schoolbus.png')
add_arg('image_path', str, 'input/cat_example.png', 'given the image path, e.g., input/schoolbus.png')
add_arg('num_iterations', int, 1, 'iter num for hsja')
add_arg('norm', str, 'l2', 'choose between [l2, linf]')
args = parser.parse_args()
print_arguments(args)

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def read_target_image():
    """

    Returns:

    """
    img_ori = cv2.imread(args.target_image)
    im = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224))
    im = (im.T / 255).astype(np.float32)

    orig = img_ori[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = old_div((img - mean), std)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img_tensor = paddle.to_tensor(img, dtype='float32', stop_gradient=False)
    return im, img_tensor

def main(orig):
    """

    Args:
        orig: input image, type: ndarray, size: h*w*c
        method: denoising method
    Returns:

    """

    # Define what device we are using
    logging.info("CUDA Available: {}".format(paddle.is_compiled_with_cuda()))
    img = orig.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = old_div((img - mean), std)

    img = img.transpose(2, 0, 1)
    C, H, W = img.shape
    img = np.expand_dims(img, axis=0)
    img = paddle.to_tensor(img, dtype='float32', stop_gradient=False)

    # Initialize the network
    model = paddle.vision.models.resnet101(pretrained=True, num_classes=1000)
    model.eval()
    # init a paddle model
    paddle_model = PaddleBlackBoxModel(
        [model],
        [1],
        (0, 1),
        mean=mean,
        std=std,
        input_channel_axis=0,
        input_shape=(3, 224, 224),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=1000)

    predict = model(img)[0]
    label = np.argmax(predict)
    img = np.squeeze(img)
    inputs = img
    labels = label

    # Read the labels file for translating the labelindex to english text
    with open('../../../Robustness/perceptron/utils/labels.txt') as info:
        imagenet_dict = eval(info.read())
    print(bcolors.CYAN + "input image label: {}".format(imagenet_dict[label]) + bcolors.ENDC)
    print(bcolors.CYAN + "input image label: {}".format(label) + bcolors.ENDC)
    adversary = Adversary(inputs.numpy(), labels)
    # non-targeted attack
    attack_config = {"steps": 100}
    print()
    if args.target_image:
        target_image, target_image_tensor = read_target_image()
        predict = model(target_image_tensor)[0]
        tlabel = np.argmax(predict)
        print(bcolors.CYAN + "target image label: {}".format(imagenet_dict[tlabel]) + bcolors.ENDC)
        print(bcolors.CYAN + "target image label: {}".format(tlabel) + bcolors.ENDC)        
        params = {
            'target_label': tlabel,
            'target_image': target_image,
            'constraint': args.norm,
            'num_iterations': args.num_iterations
        }        
        attack = HopSkipJumpAttack(paddle_model, params=params)
        adversary.set_status(is_targeted_attack=True, target_label=tlabel)
    else:
        params = {
            'target_label': None,
            'target_image': None,
            'constraint': args.norm,
            'num_iterations': args.num_iterations
        }        
        attack = HopSkipJumpAttack(paddle_model, params=params)

    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(bcolors.RED + "HSJA succeeded, adversarial_label: {}".format( \
            imagenet_dict[adversary.adversarial_label]) + bcolors.ENDC)
        print(bcolors.RED + "HSJA succeeded, adversarial_label: {}".format( \
            adversary.adversarial_label) + bcolors.ENDC)
        adv = adversary.adversarial_example
        adv = np.squeeze(adv)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        adv_cv = np.copy(adv)
        adv_cv = adv_cv[..., ::-1]  # RGB to BGR
        cv2.imwrite('output/img_adv_hsja.png', adv_cv)
        show_images_diff(orig, labels, adv, adversary.adversarial_label)
    else:
        print('attack failed')


if __name__ == '__main__':
    # read image
    orig = cv2.imread(args.image_path)
    orig = orig[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    # denoise
    main(orig)


