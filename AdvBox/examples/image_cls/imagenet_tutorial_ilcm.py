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
ILCM tutorial on imagenet using the attack tool.
ILCM method supports targeted attack.
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

from adversary import Adversary
from attacks.gradient_method import ILCM
from models.whitebox import PaddleWhiteBoxModel
from examples.utils import add_arguments, print_arguments, show_images_diff

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('target', int, -1, "target class.")

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def main(image_path):
    """

    Args:
        image_path: path of image to be test

    Returns:

    """
    # parse args
    args = parser.parse_args()
    print_arguments(args)

    # Define what device we are using
    logging.info("CUDA Available: {}".format(paddle.is_compiled_with_cuda()))

    orig = cv2.imread(image_path)[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = old_div((img - mean), std)
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)
    img = paddle.to_tensor(img, dtype='float32', stop_gradient=False)

    # Initialize the network
    model = paddle.vision.models.resnet50(pretrained=True)
    model.eval()

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

    # non-targeted attack
    attack = ILCM(paddle_model)

    predict = model(img)[0]
    print(predict.shape)
    label = np.argmax(predict)
    print("label={}".format(label)) #388

    img = np.squeeze(img)
    inputs = img
    labels = label

    print("input img shape: ", inputs.shape)
    adversary = Adversary(inputs.numpy(), labels)

    # targeted attack
    target_class = args.target
    if target_class != -1:
        tlabel = target_class
        adversary.set_status(is_targeted_attack=True, target_label=tlabel)

    attack = ILCM(paddle_model, norm='Linf', epsilon_ball=16/255, epsilon_stepsize=2/255)
    # attack = ILCM(paddle_model, norm='L2', epsilon_ball=100/255, epsilon_stepsize=100/255)

    # 设定epsilons
    attack_config = {"steps": 100}
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
        cv2.imwrite('output/img_adv_ilcm.png', adv_cv)
        show_images_diff(orig, labels, adv, adversary.adversarial_label)
    else:
        print('attack failed')

    print("ilcm attack done")


if __name__ == '__main__':
    main("input/cropped_panda.jpeg")
