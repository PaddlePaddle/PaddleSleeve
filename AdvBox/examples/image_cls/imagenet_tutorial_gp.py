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
Denoising tutorial on imagenet using the FGSM attack tool

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
from PIL import Image

from adversary import Adversary
from attacks.single_pixel_attack import SinglePixelAttack
from attacks.genetic_pixel_attack import GeneticPixelAttack
from models.whitebox import PaddleWhiteBoxModel
from examples.utils import add_arguments, print_arguments, show_images_diff
from examples.utils import bcolors

# parse args
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('target', int, -1, "target class.")
add_arg('temp', int, 100, "initial temp.")
add_arg('max_pixels', int, 40, "the maximum number of pixels allowed to be changed.")
add_arg('max_gen', int, 5000, "maximum steps")
add_arg('image_path', str, 'input/cat_example.png', 'given the image path, e.g., input/schoolbus.png')
args = parser.parse_args()
print_arguments(args)

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
    norm = paddle.vision.transforms.Normalize(mean, std)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    C,H,W = img.shape

    img = paddle.to_tensor(img, dtype='float32', stop_gradient=False)
    img = norm(img)
    img = paddle.unsqueeze(img, axis=0)

    # Initialize the network
    model = paddle.vision.models.resnet50(pretrained=True)
    model.eval()

    predict = model(img)[0]
    print (predict.shape)
    label = np.argmax(predict)
    print("label={}".format(label))
    img = np.squeeze(img)
    inputs = img
    labels = label
    print("input img shape: ", inputs.shape)

    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [model],
        [1],
        (0, 1),
        mean=mean,
        std=std,
        input_channel_axis=0,
        input_shape=(C, H, W),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=1000)

    adversary = Adversary(inputs.numpy(), labels)

    # non-targeted attack
    attack_config = {"mutation_rate": 0.01, "population": 20, "max_gen": args.max_gen, "temp": args.temp, "max_pixels": args.max_pixels, "target": args.target}
    attack = GeneticPixelAttack(paddle_model, **attack_config)
    # targeted attack not supported for now

    adversary = attack(adversary)

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
        cv2.imwrite('output/img_adv_gp.png', adv_cv)
        show_images_diff(orig, labels, adv, adversary.adversarial_label)
    else:
        print('attack failed')

if __name__ == '__main__':
    main('input/cat_example.png')
