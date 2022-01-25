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
unittest for dlg attack modulus
"""
from __future__ import print_function

import unittest

import time

import attack
import paddle
import numpy as np
from paddle import nn

from privbox.inversion import DLGInversionAttack

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
        label = np.random.randint(0, 2, 1).reshape([1, 1]).astype('int32')
        return image, label

    def __len__(self):
        return self.data_size


class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(2, 2)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.linear(x)
        return y
        

class TestDLG(unittest.TestCase):
    """
    Test DLG Attack modulus
    """
    
    def test_main(self):
        net = SimpleNet()

        dataset = SimpleDataset(100, [2], 2)
        dataload = paddle.io.DataLoader(dataset)

        data = dataload().next()

        pre = net(data[0])

        label = paddle.nn.functional.one_hot(data[1], 2).astype('float32')

        loss = paddle.nn.functional.mse_loss(pre, label, reduction='none')


        grad = paddle.grad(loss, net.parameters())

        attack = DLGInversionAttack(net, grad, data[0].shape, data[1].shape)

        params = {"learning_rate": 0.2, "attack_epoch":2, "window_size":100, "return_epoch":1}
        attack.set_params(**params)

        result = attack.reconstruct()


if __name__ == '__main__':
    unittest.main()
