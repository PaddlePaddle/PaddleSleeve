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
SinglePixelAttack tutorial on mnist using advbox tool.
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
from attacks.single_pixel_attack import SinglePixelAttack
from mnist_cnn_model import CNNModel
from models.blackbox import PaddleBlackBoxModel
from examples.utils import add_arguments, print_arguments

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
    mean = [127.5]
    std = [127.5]

    # normalize
    transform = Compose([Normalize(mean=mean,
                                   std=std,
                                   data_format='CHW')])

    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1)

    model = CNNModel()

    # init a paddle black box model
    paddle_model = PaddleBlackBoxModel(
        [model],
        [1],
        (0, 255),
        mean=mean,
        std=std,
        input_channel_axis=0,
        input_shape=(1, 28, 28),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=10)

    # 形状为[1,28,28] channel_axis=0  形状为[28,28,1] channel_axis=2
    attack = SinglePixelAttack(paddle_model)
    attack_config = {"max_pixels": 28 * 28}

    model_path = "finetuing_cnn/mnist_cnn.pdparams"
    if os.path.exists(model_path):
        para_state_dict = paddle.load(model_path)
        model.set_dict(para_state_dict)
        print('Loaded trained params of model successfully')
    else:
        print("model path not ok!")
        raise ValueError('The model_path is wrong: {}'.format(model_path))

    model.eval()
    #batch_size = 1
    total_count = 0
    fooling_count = 0
    TOTAL_NUM = 20
    for batch_id, data in enumerate(test_loader()):
        total_count += 1
        x_data = data[0]
        y_data = data[1]
        #print(x_data[0]) #-1 ~ 1
        #print(x_data[0].shape)#[1, 28, 28]

        #print (x_data.shape, type(x_data)) #NCHW format
        #print (y_data.shape, type(y_data))
        predicts = model(x_data)
        #print (predicts.shape)#[64,10]
        orig_label = np.argmax(predicts[0])
        #print ("=====pred label: ", np.argmax(predicts[0]))#[64,10]#2

        #attack
        img = np.reshape(x_data.numpy(), [1, 28, 28])
        adversary = Adversary(img, int(y_data[0]))
        # SinglePixelAttack attack
        target_class = args.target
        if target_class != -1:
            tlabel = target_class
            adversary.set_status(is_targeted_attack=True, target_label=tlabel)

        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (y_data[0], adversary.adversarial_label, total_count))

            orig = adversary.original.reshape([28, 28])
            adv = adversary.adversarial_example.reshape([28, 28])
            adv = np.clip(adv, -1, 1)
            # show_images_diff(orig, adv, adversary.adversarial_label, orig_label)

        else:
            print('attack failed, original_label=%d, count=%d' %
                  (y_data[0], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break

    print("SinglePixelAttack attack done")


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
    plt.subplot(121)
    plt.title('Original: ' + str(original_label))
    plt.imshow(original_img, cmap=plt.cm.binary)
    plt.axis('off')

    plt.subplot(122)
    plt.title('Adversarial: ' + str(adversarial_label))
    plt.imshow(adversarial_img, cmap=plt.cm.binary)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("output/1pixel_attack_{}.png".format(adversarial_label))
    plt.show()


if __name__ == '__main__':
    main()
