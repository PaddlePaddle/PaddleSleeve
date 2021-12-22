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
from utility import add_arguments, print_arguments
from utility import  show_input_adv_and_denoise
from utility import bcolors

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
    model = paddle.vision.models.resnet50(pretrained=True)

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

    model.eval()
    predict = model(img)[0]
    label = np.argmax(predict)
    img = np.squeeze(img)
    inputs = img
    labels = label

    # Read the labels file for translating the labelindex to english text
    with open('../../../Robustness/perceptron/utils/labels.txt') as info:
        imagenet_dict = eval(info.read())
    print(bcolors.CYAN + "input image label: {}".format(imagenet_dict[label]) + bcolors.ENDC)

    adversary = Adversary(inputs.numpy(), labels)

    # non-targeted attack
    attack_config = {"mutation_rate": 0.01, "population": 20, "max_gen": args.max_gen, "temp": args.temp, "max_pixels": args.max_pixels, "target": args.target}
    attack = GeneticPixelAttack(paddle_model, **attack_config)
    # targeted attack not supported for now

    adversary = attack(adversary)

    if adversary.is_successful():
        print(bcolors.RED + "Genetic Pixel attack succeeded, adversarial_label: {}".format(\
            imagenet_dict[adversary.adversarial_label]) + bcolors.ENDC)

        adv = adversary.adversarial_example
        adv = np.squeeze(adv)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        de_adv_input = np.copy(adv).transpose(2, 0, 1) / 255

    else:
        print('attack failed')

def ndarray2opencv(img):
    """
    Convert ndarray to opencv image
    :param img: the input image, type: ndarray
    :return: an opencv image
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = np.transpose(img, (1, 2, 0))
    img = (img * std) + mean
    img = img - np.floor(img)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img



if __name__ == '__main__':
    # read image
    orig = cv2.imread(args.image_path)
    # cv2.imwrite('test_orig.png', orig)
    orig = orig[..., ::-1]
    # denoise
    main(orig)

    # # ***** find the proper image sample from the mini-imagenet test-set *****
    # import pickle
    # with open('input/mini-imagenet-cache-test.pkl','rb') as f:
    #     data=pickle.load(f)
    # imagedata = data['image_data']
    # for i in range(imagedata.shape[0]):
    #     # original_image = np.copy(imagedata[i+9034])
    #     # original_image = original_image[...,  ::-1]  # RGB to BGR
    #     # cv2.imwrite('input/schoolbus.png', original_image)
    #     # break
    #     main(imagedata[i+9000])
