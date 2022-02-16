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
unittest for metric modulus
"""
from __future__ import print_function

import unittest
import time

import paddle
import numpy as np
from paddle import nn

from privbox.inference.membership_inference import BaselineMembershipInferenceAttack


class TestBaselineMembershipInferenceAttack(unittest.TestCase):
    """
    Test BaselineMembershipInferenceAttack
    """
    
    def test_main(self):
        """
        main test case for BaselineMembershipInferenceAttack
        """
        attack = BaselineMembershipInferenceAttack()
        one = paddle.ones([1], dtype="int32")
        zero = paddle.zeros([1], dtype="int32")
        result = attack.infer([one, one])
        self.assertEqual(result.numpy().item(), 1)

        result = attack.infer([one, zero])
        self.assertEqual(result.numpy().item(), 0)


if __name__ == '__main__':
    unittest.main()
