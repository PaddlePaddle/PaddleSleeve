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
Data augmentation demo. Train resnet34 on Cifar10
"""

import os
import sys
sys.path.append("../..")

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from perceptron.augmentations.augment import SerialAugment

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def main(dataaug=False):
    data_augment = SerialAugment(transforms=[{'Rotate': {}},
                                             {'Translation': {}},
                                             {'HueSaturation': {}},
                                             {'GridDistortion': {}}],
                                 format='HWC', bound=(0, 255))
    if dataaug:
        train_transform = T.Compose([data_augment,
                               T.Normalize(mean=[125.31, 122.95, 113.86], std=[62.99, 62.08, 66.7], data_format='HWC'),
                               T.Transpose(order=(2, 0, 1))])
    else:
        train_transform = T.Compose([T.Normalize(mean=[125.31, 122.95, 113.86], std=[62.99, 62.08, 66.7], data_format='HWC'),
                                     T.Transpose(order=(2, 0, 1))])

    test_transform = T.Compose([T.Normalize(mean=[125.31, 122.95, 113.86], std=[62.99, 62.08, 66.7], data_format='HWC'),
                                T.RandomRotation(30),
                                T.HueTransform(0.2),
                                T.RandomCrop(size=32, padding=4),
                                T.Transpose(order=(2, 0, 1))])

    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=train_transform, backend='cv2')
    val_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=test_transform, backend='cv2')

    model = paddle.vision.models.resnet34(pretrained=False, num_classes=10)
    model = paddle.Model(model)
    BATCH_SIZE = 128
    train_loader = paddle.io.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = paddle.io.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    learning_rate = 0.001
    loss_fn = paddle.nn.CrossEntropyLoss()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    model.prepare(optimizer=opt, loss=loss_fn, metrics=paddle.metric.Accuracy())

    model.fit(train_loader, test_loader, batch_size=128, epochs=20, eval_freq=5, verbose=1)
    model.evaluate(test_loader, verbose=1)


if __name__ == '__main__':
    print('(base) ~/Users/paddlesleeve/Robustness# vim perceptron/augmentations/cifar10_dataaug_tutorial.py ')
    main(True)
    main(False)
