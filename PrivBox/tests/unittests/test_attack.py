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
unittest for attack modulus
"""
from __future__ import print_function

import unittest

import privbox.attack as attack
from privbox.metrics import Accuracy
import paddle
import numpy as np

class ImplAttack(attack.Attack):
    """
    A simple attack impl
    """
    params = ["key0", "key1", "key2"]
    def set_params(self, **kwargs):
        super().set_params(**kwargs)
    
    def evaluate(self, target, result, metrics, **kwargs):
        super().evaluate(target, result, metrics, **kwargs)

class TestAttack(unittest.TestCase):
    """
    Test Attack modulus
    """
    attack = ImplAttack()

    def test_set_params(self):
        params = {"key0": 1, "key1":"1", "key3":2}
        self.attack.set_params(**params)
        self.assertTrue(self.attack.key0 == 1)
        self.assertTrue(self.attack.key1 == "1")
        self.assertFalse(hasattr(self.attack, 'key2'))
        self.assertFalse(hasattr(self.attack, 'key3'))

    def test_evaluate(self):
        target = paddle.to_tensor(np.array([1]))
        result = paddle.to_tensor(np.array([1]))
        
        # no exception
        self.attack.evaluate(target, result, [Accuracy()])


if __name__ == '__main__':
    unittest.main()
