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

from adversary import Adversary
from denoising import Denoising
from denoisers.denoising_method import GaussianBlur
from denoisers.denoising_method import MedianBlur
from denoisers.denoising_method import MeanFilter
from denoisers.denoising_method import BilateralFilter
from denoisers.denoising_method import BoxFilter
from denoisers.denoising_method import PixelDeflection
from denoisers.denoising_method import JPEGCompression
from denoisers.denoising_method import DCTCompression
from denoisers.denoising_method import PCACompression
from denoisers.denoising_method import GaussianNoise
from denoisers.denoising_method import SaltPepperNoise
from denoisers.denoising_method import ResizePadding
from denoisers.denoising_method import FeatureSqueezing
from attacks.gradient_method import FGSMT
from attacks.gradient_method import FGSM
from models.whitebox import PaddleWhiteBoxModel
from utility import add_arguments, print_arguments
from utility import  show_input_adv_and_denoise
from utility import bcolors

# parse args
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('target', int, -1, "target class.")
add_arg('method', str, 'ResizePadding', 'given the denoising method, e.g., GaussianBlur')
add_arg('image_path', str, 'input/hourglass.png', 'given the image path, e.g., input/schoolbus.png')
args = parser.parse_args()
print_arguments(args)
# MedianBlur 'input/vase.png'
# MeanFilter 'input/lion.png'
# BoxFilter 'input/hourglass.png'
# BilateralFilter 'input/crate.png'
# PixelDeflection 'input/malamute.png'
# JPEGCompression 'input/schoolbus.png'
# DCTCompression 'input/vase.png'
# PCACompression 'input/vase.png'
# GaussianNoise 'input/schoolbus.png'
# SaltPepperNoise 'input/schoolbus.png'
# ResizePadding 'input/schoolbus.png'

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)

denoisers = {
    'GaussianBlur': GaussianBlur,
    'MedianBlur': MedianBlur,
    'MeanFilter': MeanFilter,
    'BoxFilter': BoxFilter,
    'BilateralFilter': BilateralFilter,
    'PixelDeflection': PixelDeflection,
    'JPEGCompression': JPEGCompression,
    'DCTCompression': DCTCompression,
    'PCACompression': PCACompression,
    'GaussianNoise': GaussianNoise,
    'SaltPepperNoise': SaltPepperNoise,
    'ResizePadding': ResizePadding,
    'FeatureSqueezing': FeatureSqueezing,
}

def main(orig):
    """

    Args:
        orig: input image, type: ndarray, size: h*w*c
        method: denoising method
    Returns:

    """

    Denoiser = denoisers[args.method]

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
    attack = FGSM(paddle_model)

    # targeted attack
    target_class = args.target
    if target_class != -1:
        tlabel = target_class
        adversary.set_status(is_targeted_attack=True, target_label=tlabel)

        attack = FGSMT(paddle_model, norm='Linf',
                       epsilon_ball=100 / 255, epsilon_stepsize = 100 / 255)

    # 设定epsilons
    adversary = attack(adversary)

    if adversary.is_successful():
        print(bcolors.RED + "FGSM attack succeeded, adversarial_label: {}".format(\
            imagenet_dict[adversary.adversarial_label]) + bcolors.ENDC)

        adv = adversary.adversarial_example
        adv = np.squeeze(adv)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        de_adv_input = np.copy(adv).transpose(2, 0, 1) / 255

        # denoising start
        denoising_config = {"steps": 10}
        denoise_method = Denoiser(paddle_model)

        # init the adversary image to be denosied
        de_adv = Denoising(de_adv_input, adversary.adversarial_label, labels)

        de_adv = denoise_method(de_adv, **denoising_config)
        if de_adv.is_successful():
            denoising_image = de_adv.denoising_sample
            denoising_image = np.squeeze(denoising_image)
            denoising_image = denoising_image.transpose(1, 2, 0)
            denoising_image = denoising_image * 255.0
            denoising_image = np.clip(denoising_image, 0, 255).astype(np.uint8)
            print(bcolors.GREEN + args.method + ' denoising succeeded' + bcolors.ENDC)

            # init the input image to be denosied
            ori_image = orig.copy().astype(np.float32)
            ori_image /= 255.0
            ori_image = ori_image.transpose(2, 0, 1)
            de_input = Denoising(ori_image, labels, labels)
            de_input = denoise_method(de_input, **denoising_config)
            if de_input.is_successful():
                denoising_input_image = de_input.denoising_sample
                denoising_input_image = np.squeeze(denoising_input_image)
                denoising_input_image = denoising_input_image.transpose(1, 2, 0)
                denoising_input_image = denoising_input_image * 255.0
                denoising_input_image = \
                    np.clip(denoising_input_image, 0, 255).astype(np.uint8)
                print(
                    bcolors.GREEN + args.method + \
                    ' denoising doesn\'t change the label of the input image' + \
                    bcolors.ENDC)
                show_input_adv_and_denoise(orig, adv, \
                                           denoising_image, denoising_input_image, \
                                           imagenet_dict[label], \
                                           imagenet_dict[adversary.adversarial_label], \
                                           imagenet_dict[de_adv.denoising_label], \
                                           imagenet_dict[de_input.denoising_label], \
                                           'Input', 'Adversary', \
                                           'Adv-Denoise', 'Input-Denoise', \
                                           args.method)
            else:
                print(bcolors.CYAN + args.method + \
                      ' denoising changes the label of the input image' + \
                      bcolors.ENDC)
        else:
            print(bcolors.CYAN + args.method + ' denoising on AE failed' + bcolors.ENDC)
    else:
        print('attack failed')


if __name__ == '__main__':
    # read image
    orig = cv2.imread(args.image_path)[..., ::-1]
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
