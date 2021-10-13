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
This module provides an example of how to use rounding, labeling and topk post-procesing defenses.
"""

from __future__ import print_function

import sys
import os

import numpy
import numpy as np

import paddle
from paddle.vision.transforms import Compose, Normalize

from rounding import RoundingNet
from labeling import LabelingNet
from topk import TopKNet


class LinearNet(paddle.nn.Layer):
    """
    Define a Linear Network for MNIST
    """
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear = paddle.nn.Linear(28 * 28, 10)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.linear(x.reshape((-1, 28 * 28)))
        return y


def train():
    """
    train model, then construct post-processing network
    """

    # load mnist data
    transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

    # define a linear network and train it
    net = LinearNet()
    model = paddle.Model(net)

    model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
                   paddle.nn.CrossEntropyLoss(),
                   paddle.metric.Accuracy())

    model.fit(train_dataset, test_dataset, batch_size=128, epochs=1)

    origin_predict = model.network(paddle.to_tensor(test_dataset[0][0]))
    print("origin network output: ", origin_predict)

    # rounding the trained network
    rounding_net = RoundingNet(model.network, 2, False, axes=[0], starts=[0], ends=[1])
    rounding_predict = rounding_net(paddle.to_tensor(test_dataset[0][0]))
    print("rounding network output (precision = 2): ", rounding_predict)

    # labeling the trained network
    label_net = LabelingNet(model.network)
    label_predict = label_net(paddle.to_tensor(test_dataset[0][0]))
    print("label network output (i.e., label indices): ", label_predict)

    # top-k the trained network
    topk_net = TopKNet(model.network, 3, False)
    label_predict = topk_net(paddle.to_tensor(test_dataset[0][0]))
    print("topk network output (i.e., top-k (values, indices) pairs): ", label_predict)


if __name__ == "__main__":
    train()
