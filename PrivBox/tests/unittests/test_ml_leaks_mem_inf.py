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
unittest for ML-Leaks attack modulus
"""
from __future__ import print_function

import unittest

import time

import privbox.attack
import paddle
import numpy as np
from paddle import nn

from privbox.inference.membership_inference import MLLeaksMembershipInferenceAttack

class SimpleDataset(paddle.io.Dataset):
    """
    Simple Dataset for test
    """
    def __init__(self, data_size, data_shape, label_size):
        self.data_size = data_size
        self.data_shape = data_shape
        self.label_size = label_size

    def __getitem__(self, idx):
        image = np.random.random(self.data_shape).astype('float32')
        label = np.random.randint(0, 2, 1).reshape([1]).astype('int64')
        return image, label

    def __len__(self):
        return self.data_size


class SimpleNet(nn.Layer):
    """
    Simple Net for test
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(2, 2)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.linear(x)
        return y
        

class TestMLLeaks(unittest.TestCase):
    """
    Test ML-Leaks Attack modulus
    """
    
    def test_normal(self):
        """
        Used trained shadow model
        """
        # assume net have been trained
        net = SimpleNet()
        dataset = SimpleDataset(100, [2], 2)

        attack = MLLeaksMembershipInferenceAttack(net, [dataset, dataset])

        attack_params = {"batch_size": 1, "shadow_epoch": 1,
                     "classifier_epoch": 1, "topk": 2,
                     "shadow_lr":0.01, "classifier_lr": 0.01}
        attack.set_params(**attack_params)

        data = paddle.rand([2, 2])
        pred = net(data)

        attack.infer(pred)


if __name__ == '__main__':
    unittest.main()
