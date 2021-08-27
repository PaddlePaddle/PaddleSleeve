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
Denoising tutorial on imagenet using the FGSM attack

"""
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../..")
import logging
import argparse
import functools
import numpy as np
import paddle
from adversary import Adversary
from denoising import Denoising
from denoisers.denoising_method import GaussianBlur
from denoisers.denoising_method import MedianBlur
from denoisers.denoising_method import MeanFilter
from denoisers.denoising_method import BilateralFilter
from denoisers.denoising_method import BoxFilter
from denoisers.denoising_method import PixelDeflection
from denoisers.denoising_method import JPEGCompression
from denoisers.denoising_method import DCTCompress
from denoisers.denoising_method import PCACompress
from denoisers.denoising_method import GaussianNoise
from denoisers.denoising_method import SaltPepperNoise
from denoisers.denoising_method import ResizePadding
from attacks.gradient_method import FGSMT
from attacks.gradient_method import FGSM
from models.whitebox import PaddleWhiteBoxModel
from examples.utils import add_arguments, print_arguments
from miniimagenet import MINIIMAGENET
from paddle.vision.transforms import Compose, Normalize
from tqdm import tqdm
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
print(paddle.__version__)
print(paddle.in_dynamic_mode())

# parse args
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('target', int, -1, "target class.")
add_arg('method', str, 'SaltPepperNoise', 'given the denoising method, e.g., GaussianBlur')
add_arg('dataset_path', str, 'input/mini-imagenet-cache-test.pkl', \
        'given the dataset path, e.g., input/mini-imagenet-cache-test.pkl')
add_arg('label_path', str, 'input/mini_imagenet_test_labels.txt', \
        'given the dataset path, e.g., input/mini_imagenet_test_labels.txt')
add_arg('mode', str, 'test', 'specify the dataset, e.g., train, test, or val')
args = parser.parse_args()
print_arguments(args)

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)

# Define what device we are using
logging.info("CUDA Available: {}".format(paddle.is_compiled_with_cuda()))

# Dict of denoising methods
denoisers = {
    'GaussianBlur': GaussianBlur,
    'MedianBlur': MedianBlur,
    'MeanFilter': MeanFilter,
    'BoxFilter': BoxFilter,
    'BilateralFilter': BilateralFilter,
    'PixelDeflection': PixelDeflection,
    'JPEGCompression': JPEGCompression,
    'DCTCompress': DCTCompress,
    'PCACompress': PCACompress,
    'GaussianNoise': GaussianNoise,
    'SaltPepperNoise': SaltPepperNoise,
    'ResizePadding': ResizePadding,
}

def test():
    """
    Perform attack and denoise on the given dataset, and evaluate the performance.
    Args:
    Returns:

    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Set the classification network
    model = paddle.vision.models.resnet50(pretrained=True)

    # Set the attack model
    loss_fn = paddle.nn.CrossEntropyLoss()
    paddle_model = PaddleWhiteBoxModel(
        [model],
        [1],
        loss_fn,
        (-3, 3),
        channel_axis=3,
        nb_classes=1000)
    # paddle_model = PaddleWhiteBoxModel(
    #     [model],
    #     [1],
    #     (0, 1),
    #     mean=mean,
    #     std=std,
    #     input_channel_axis=0,
    #     input_shape=(3, 224, 224),
    #     loss=paddle.nn.CrossEntropyLoss(),
    #     nb_classes=1000)

    # Set the attack method
    target_class = args.target
    if target_class == -1:
        # untargeted attack
        attack = FGSM(paddle_model)
    else:
        # targeted attack
        tlabel = target_class
        attack = FGSMT(paddle_model)

    # Set the denoising method
    Denoiser = denoisers[args.method]
    denoise_method = Denoiser(paddle_model)

    # Set the testing set
    transform = Compose([Normalize(mean, std, data_format='CHW')])
    dataset_path = args.dataset_path
    label_path = args.label_path
    mode = args.mode
    test_dataset = MINIIMAGENET(dataset_path = dataset_path, \
                                label_path = label_path, \
                                mode = mode, \
                                transform = transform)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Start testing
    ori_pos = 0
    total = 0
    adv = 0
    de_adv_pos = 0
    de_ori_neg = 0
    ORI_ACC = 0
    AE_ACC = 0
    DE_AE_ACC = 0
    DE_ORI_ACC = 0
    loop = tqdm(enumerate(test_loader), total = len(test_loader), ncols = 150, miniters = 0)
    for id, data in loop:
        img = data[0]
        gt_label = data[1]

        # Get the prediction of the original model
        model.eval()
        predict = model(img)
        label = np.argmax(predict)
        total = total + 1

        # Discard the misclassified image
        if label == gt_label:
            ori_pos = ori_pos + 1
            # Start attack and denoise
            img = np.squeeze(img)
            inputs = img
            labels = label

            # Set the attack parameters
            attack_config = {"epsilons": 0.3, "epsilon_steps": 10, "steps": 1}
            adversary = Adversary(inputs.numpy(), labels)
            if target_class != -1:
                adversary.set_status(is_targeted_attack=True, target_label=tlabel)
            # Set the denoising paramters
            denoising_config = {"steps": 10}

            # Start attack method
            adversary = attack(adversary, **attack_config)
            if adversary.is_successful():
                adv = adv + 1
                # Start AE denoising
                adversary_image = adversary.adversarial_example
                adversary_image = np.squeeze(adversary_image)
                adversary_image = adversary_image.transpose(1, 2, 0)
                adversary_image = (std * adversary_image) + mean
                adversary_image = adversary_image.transpose(2, 0, 1)
                denoising = Denoising(adversary_image, adversary.adversarial_label, labels)
                denoising = denoise_method(denoising, **denoising_config)
                if denoising.is_successful():
                    de_adv_pos = de_adv_pos + 1

            # Start original image denoising
            ori = inputs.numpy().transpose(1, 2, 0)
            ori = (std * ori) + mean
            ori = ori.transpose(2, 0, 1)
            denoising_ori = Denoising(ori, labels, labels)
            denoising_ori = denoise_method(denoising_ori, **denoising_config)
            if not denoising_ori.is_successful():
                de_ori_neg = de_ori_neg + 1

            # Update accuracy info
            ORI_ACC = ori_pos / total
            AE_ACC = (ori_pos - adv) / total
            DE_AE_ACC = (ori_pos - adv + de_adv_pos) / total
            DE_ORI_ACC = (ori_pos - de_ori_neg) / total
            loop.set_postfix(OrderedDict(ORI_ACC = '{:.3f}'.format(ORI_ACC), \
                                         AE_ACC = '{:.3f}'.format(AE_ACC), \
                                         DE_AE_ACC = '{:.3f}'.format(DE_AE_ACC), \
                                         DE_ORI_ACC = '{:.3f}'.format(DE_ORI_ACC)))


if __name__ == '__main__':
    test()
