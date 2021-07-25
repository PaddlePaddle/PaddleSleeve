# encoding:utf-8

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

"""
The tutorial of serial ensemble models blackbox attack based on FGSM.
"""

from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../..")

from past.utils import old_div
import logging
logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import argparse
import cv2
import functools
import numpy as np
import paddle
print(paddle.__version__)
print(paddle.in_dynamic_mode())
# Define what device we are using
logging.info("CUDA Available: {}".format(paddle.is_compiled_with_cuda()))

from adversary import Adversary
from attacks.gradient_method import FGSMT
from models.whitebox import PaddleWhiteBoxModel
from utility import add_arguments, print_arguments, show_images_diff

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('target', int, -1, "target class.")

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def predict(image_path, model):
    """
    Get the result of the classification model.
    Args:
        image_path: path of the image
        model: the classification model
    Returns: 
        the classification result label
    """
    model.eval()
    orig = cv2.imread(image_path)[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = old_div((img - mean), std)
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)
    img = paddle.to_tensor(img, dtype='float32',
                           place=paddle.get_device(), stop_gradient=False)

    predict_result = model(img)[0]
    label = np.argmax(predict_result)
    return label


def target_attack_fgsm(input_image_path, output_image_path, model, tlabel):
    """
    Use iterative target FGSM attack for a model.
    Args:
        input_image_path: the path of the input image 
        output_image_path: the path of the output image 
        model: the image classification model 
        tlabel: the target label
    Returns: 
    """
    label = predict(input_image_path, model)
    print("original label={}".format(label))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    orig = cv2.imread(input_image_path)[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)

    img /= 255.0
    img = old_div((img - mean), std)
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)
    img = paddle.to_tensor(img, dtype='float32',
                           place=paddle.get_device(), stop_gradient=False)

    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [model],
        [1],
        (0, 1),
        mean=mean,
        std=std,
        input_channel_axis=0,
        input_shape=(3, 224, 224),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=1000)

    inputs = np.squeeze(img)
    adversary = Adversary(inputs.numpy(), label)
    adversary.set_status(is_targeted_attack=True, target_label=tlabel)

    attack = FGSMT(paddle_model)
    # 设定epsilons
    attack_config = {"epsilons": 0.5, "epsilon_steps": 10, "steps": 50}
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
            % (adversary.adversarial_label))

        adv = adversary.adversarial_example
        adv = np.squeeze(adv)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        adv_cv = np.copy(adv)
        adv_cv = adv_cv[..., ::-1]  # RGB to BGR
        cv2.imwrite(output_image_path, adv_cv)
        # show_images_diff(orig, labels, adv, adversary.adversarial_label)
    else:
        print('attack failed')

    print("FGSMT attack done")


def main():
    """
    Use several models' logits to generate a adversary example for blackbox attack.
    Args:
        image_path: the path of image to be tested
    Returns:
    """
    input_path = "input/cat_example.png"
    output_path = "output/img_adv.png"

    # parse args
    args = parser.parse_args()
    print_arguments(args)
    target_label = args.target
    if target_label == -1:
        print("ERROR: need a target")
        sys.exit(0)

    attack_model = paddle.vision.models.resnet50(pretrained=True)
    target_attack_fgsm(input_path, output_path, attack_model, target_label)
    label = predict(output_path, attack_model)
    print("resnet50 adv label={}".format(label))

    attack_model = paddle.vision.models.mobilenet_v1(pretrained=True)
    target_attack_fgsm(output_path, output_path, attack_model, target_label)
    label = predict(output_path, attack_model)
    print("mobilenet_v1 adv label={}".format(label))

    attack_model = paddle.vision.models.resnet18(pretrained=True)
    target_attack_fgsm(output_path, output_path, attack_model, target_label)
    label = predict(output_path, attack_model)
    print("resnet18 adv label={}".format(label))

    # victim model
    victim_model = paddle.vision.models.vgg16(pretrained=True)
    label = predict(output_path, victim_model)
    print("victim vgg16 predict label={}".format(label))


if __name__ == '__main__':
    main()
