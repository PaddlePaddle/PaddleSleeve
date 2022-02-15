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
unittest for knockoff attack modulus
"""
from __future__ import print_function

import unittest

import time

import privbox.attack
import paddle
import numpy as np
from paddle import nn
from privbox.metrics import Accuracy

from privbox.extraction import KnockoffExtractionAttack

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
        label = np.random.randint(0, 2, 1).reshape([1, 1]).astype('int64')
        return image, label

    def __len__(self):
        return self.data_size


class SimpleNet(nn.Layer):
    """
    simple net for test
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
        

class TestKnockoff(unittest.TestCase):
    """
    Test Knockoff Attack modulus
    """
    
    def test_main(self):
        
        #main test
        
        net = SimpleNet()

        dataset = SimpleDataset(100, [2], 2)

        print("data0: ", dataset[0])

        attack = KnockoffExtractionAttack(net, net)

        # set attack params
        params = {"policy": "random", "has_label": False, "reward": "all",
                "num_labels": 2, "num_queries": 2,
                "knockoff_batch_size": 1, "knockoff_epochs": 1,
                "knockoff_lr": 0.1}

        attack.set_params(**params)

        # extract model
        knockoff_net = attack.extract(dataset)

        # evaluate attack
        kwargs_dataset = {"test_dataset": dataset}
        attack.evaluate(net, knockoff_net, [Accuracy()], **kwargs_dataset)
    
    def test_adaptive_policy(self):
        """
        main test
        """
        net = SimpleNet()

        dataset = SimpleDataset(100, [2], 2)

        attack = KnockoffExtractionAttack(net, net)

        # set attack params
        params = {"policy": "adaptive", "has_label": True, "reward": "all",
                "num_labels": 2, "num_queries": 1,
                "knockoff_batch_size": 1, "knockoff_epochs": 1,
                "knockoff_lr": 0.1}

        attack.set_params(**params)
        print("data0: ", dataset[0][1][0])

        # extract model
        knockoff_net = attack.extract(dataset)

        # evaluate attack
        kwargs_dataset = {"test_dataset": dataset}
        attack.evaluate(net, knockoff_net, [Accuracy()], **kwargs_dataset)


if __name__ == '__main__':
    unittest.main()
