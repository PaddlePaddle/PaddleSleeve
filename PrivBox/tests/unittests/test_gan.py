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
unittest for gan attack modulus
"""
from __future__ import print_function

import unittest

import time

import privbox.attack
import paddle
import numpy as np
from paddle import nn

from privbox.inversion import GANInversionAttack

class SimpleDataset(paddle.io.Dataset):
    """
    Simple dataset for test
    """
    def __init__(self, data_size, data_shape, label_size):
        self.data_size = data_size
        self.data_shape = data_shape
        self.label_size = label_size

    def __getitem__(self, idx):
        image = np.random.random(self.data_shape).astype('float32')
        label = np.random.randint(0, 2, 1).reshape([1]).astype('int32')
        return image, label

    def __len__(self):
        return self.data_size


class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, x):
        return x
        

class TestGAN(unittest.TestCase):
    """
    Test GAN Attack modulus
    """
    dataset = SimpleDataset(100, [2], 2)
    dataload = paddle.io.DataLoader(dataset)
    def test_init(self):
        gen = SimpleNet()
        attack = GANInversionAttack(gen, 2.0, 0, 1, self.dataload)
        gan_params = {"learning_rate_real": 0.1,
                    "learning_rate_fake": 0.1,
                    "learning_rate_gen": 0.1}

        self.assertRaises(ValueError, attack.set_params, **gan_params)
        # sleep for resource collect, core dump if no sleep
        time.sleep(1)
    
    def test_set_params(self):
        gen = SimpleNet()
        attack = GANInversionAttack(gen, 2, 0, 1, self.dataload)
        gan_params = {"learning_rate_real": 0.1,
                    "learning_rate_fake": 0.1,
                    "learning_rate_gen": 0.1}
        # no exception
        attack.set_params(**gan_params)

        gan_params["learning_rate_real"] = int(1)
        self.assertRaises(ValueError, attack.set_params, **gan_params)
        gan_params["learning_rate_real"] = 0.1

        gan_params["learning_rate_fake"] = int(1)
        self.assertRaises(ValueError, attack.set_params, **gan_params)
        gan_params["learning_rate_fake"] = 0.1

        gan_params["learning_rate_gen"] = int(1)
        self.assertRaises(ValueError, attack.set_params, **gan_params)
        gan_params["learning_rate_gen"] = 0.1
        # sleep for resource collect, core dump if no sleep
        time.sleep(1)
    
    def test_fit_with_data(self):
        gen = SimpleNet()
        disc = SimpleNet()

        attack = GANInversionAttack(gen, 2, 0, 1, self.dataload)

        gan_params = {"learning_rate_real": 0.1,
                    "learning_rate_fake": 0.1,
                    "learning_rate_gen": 0.1}
        attack.set_params(**gan_params)

        attack.fit(disc, self.dataload().next())

        epoch = {"epoch": 1}
        attack.fit(disc, self.dataload().next(), **epoch)
        # sleep for resource collect, core dump if no sleep
        time.sleep(1)
    
    def test_fit_without_data(self):
        gen = SimpleNet()
        disc = SimpleNet()
        attack = GANInversionAttack(gen, 2, 0, 1, self.dataload)

        gan_params = {"learning_rate_real": 0.1,
                    "learning_rate_fake": 0.1,
                    "learning_rate_gen": 0.1}
        attack.set_params(**gan_params)

        attack.fit(disc)

        epoch = {"epoch": 1}
        attack.fit(disc, None, **epoch)
        # sleep for resource collect, core dump if no sleep
        time.sleep(1)
    
    def test_reconstruct(self):
        gen = SimpleNet()
        disc = SimpleNet()

        attack = GANInversionAttack(gen, 2, 0, 1, self.dataload)

        gan_params = {"learning_rate_real": 0.1,
                    "learning_rate_fake": 0.1,
                    "learning_rate_gen": 0.1}
        attack.set_params(**gan_params)

        attack.fit(disc)

        result = attack.reconstruct()
        # sleep for resource collect, core dump if no sleep
        time.sleep(1)
        

if __name__ == '__main__':
    unittest.main()

