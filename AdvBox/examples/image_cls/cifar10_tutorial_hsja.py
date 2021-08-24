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
sys.path.append("../..")

import argparse
import numpy as np
import functools
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize
from paddle.vision.transforms import ToTensor

from adversary import Adversary
from examples.utils import get_best_weigthts_from_folder, add_arguments, print_arguments
from attacks.hop_skip_jump_attack import HopSkipJumpAttack

from examples.classifier.preactresnet import transform_train, transform_eval, MEAN, STD, preactresnet18
from models.blackbox import PaddleBlackBoxModel

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('target', int, -1, "target class.")

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def main():
    """
    Advbox demo which demonstrate how to use advbox.
    """
    args = parser.parse_args()
    print_arguments(args)
    mean = MEAN
    std = STD

    test_loader = paddle.vision.datasets.Cifar10(mode='test', transform=transform_eval)
    print(test_loader, type(test_loader), len(test_loader))
    print(test_loader[0][0].shape)

    model = preactresnet18()

    # init a paddle black box model
    paddle_model = PaddleBlackBoxModel(
        [model],
        [1],
        (0, 1),
        mean=mean,
        std=std,
        input_channel_axis=0,
        input_shape=(3, 32, 32),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=10)

    # 形状为[1,28,28] channel_axis=0  形状为[28,28,1] channel_axis=2
    attack = HopSkipJumpAttack(paddle_model)
    attack_config = {"steps": 100}

    model_path = get_best_weigthts_from_folder("../cifar10/preactresnet_base_tutorial_result", "base_net_")
    if os.path.exists(model_path):
        para_state_dict = paddle.load(model_path)
        model.set_dict(para_state_dict)
        print('Loaded trained params of model successfully')
    else:
        print("model path not ok!")
        raise ValueError('The model_path is wrong: {}'.format(model_path))

    model.eval()
    batch_size = 1
    total_count = 0
    fooling_count = 0
    TOTAL_NUM = 20
    for i in range(batch_size):
        total_count += 1
        data = test_loader[i]
        x_data = data[0]
        x_data = paddle.to_tensor(x_data)
        x_data = paddle.unsqueeze(x_data, axis=0)
        y_data = data[1]
        label = data[1]
        print("==from dataset label: ", label, type(label)) 

        # TODO: check preprocess.
        predicts = model(x_data)
        print (predicts.shape)#[1,10]
        orig_label = np.argmax(predicts[0])
        print ("=====pred label: ", np.argmax(predicts[0]))#[1,10]#2

        #hsja_attack
        img = np.reshape(x_data.numpy(), [3, 32, 32])
        adversary = Adversary(img, int(y_data))
        #hsja_attack
        target_class = args.target
        if target_class != -1:
            tlabel = target_class
            adversary.set_status(is_targeted_attack=True, target_label=tlabel)

        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                #% (y_data[0], adversary.adversarial_label, total_count))
                % (y_data, adversary.adversarial_label, total_count))

            orig = adversary.original.reshape([3, 32, 32])
            adv = adversary.adversarial_example.reshape([3, 32, 32])
            adv = adv.transpose(1, 2, 0)
            orig = orig.transpose(1, 2, 0)
            adv = np.clip(adv, -1, 1)
            show_images_diff(orig, adv, adversary.adversarial_label, orig_label)

        else:
            print('attack failed, original_label=%d, count=%d' %
                  (y_data, total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break

    print("HopSkipJumpAttack attack done")


def show_images_diff(original_img, adversarial_img, adversarial_label, original_label=None):
    """

    Args:
        original_img:
        adversarial_img:
        adversarial_label:
        original_label:

    Returns:

    """
    plt.figure()
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img

    print ("diff shape: ", difference.shape)
    #(-1,1)  -> (0,1)
    difference=difference / abs(difference).max()/2.0+0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("output/hsja_all.png")
    plt.show()



if __name__ == '__main__':
    main()
