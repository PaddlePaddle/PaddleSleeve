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
import sys
sys.path.append("../../")
import time

import attack
import paddle
import numpy as np
from paddle import nn

from metrics import MSE, Accuracy, AUC, Recall, Precision


class TestMSE(unittest.TestCase):
    """
    Test MSE metric
    """
    
    def test_main(self):
        """
        main test case for mse
        """
        mse = MSE()
        actual = paddle.rand([2, 2])
        expected = paddle.rand([2, 2])

        result0 = mse.compute(actual, expected)
        result1 = paddle.nn.functional.mse_loss(actual, expected)

        self.assertTrue(np.allclose(result0, result1.numpy(), atol=0.0001))


class TestAccuracy(unittest.TestCase):
    """
    Test Accuracy metric
    """
    
    def test_soft_actual(self):
        """
        soft actual input case for acc
        """
        acc = Accuracy(soft_actual=True)
        actual = paddle.rand([2, 2])
        expected = paddle.randint(0, 2, [2, 1])

        result0 = acc.compute(actual, expected)
        result1 = paddle.metric.accuracy(actual, expected)

        self.assertTrue(np.allclose(result0, result1.numpy(), atol=0.0001))

    def test_hard_actual(self):
        """
        hard actual input case for acc
        """
        acc = Accuracy(soft_actual=False, num_classes=2)
        actual = paddle.randint(0, 2, [2, 1])
        expected = paddle.randint(0, 2, [2, 1])

        result0 = acc.compute(actual, expected)
        actual = paddle.nn.functional.one_hot(actual, num_classes=2)
        result1 = paddle.metric.accuracy(actual, expected)

        self.assertTrue(np.allclose(result0, result1.numpy(), atol=0.0001))


class TestAUC(unittest.TestCase):
    """
    Test AUC metric
    """
    
    def test_soft_actual(self):
        """
        soft actual input case for AUC
        """
        auc = AUC(soft_actual=True)
        actual = paddle.rand([2, 2])
        expected = paddle.randint(0, 2, [2, 1])

        result0 = auc.compute(actual, expected)
        paddle_auc = paddle.metric.Auc()
        paddle_auc.update(actual, expected)

        self.assertTrue(np.allclose(result0, paddle_auc.accumulate(), atol=0.0001))

    def test_hard_actual(self):
        """
        hard actual input case for AUC
        """
        auc = AUC(soft_actual=False)
        actual = paddle.randint(0, 2, [2, 1])
        expected = paddle.randint(0, 2, [2, 1])

        result0 = auc.compute(actual, expected)
        actual = paddle.nn.functional.one_hot(actual.reshape([2]), num_classes=2)
        paddle_auc = paddle.metric.Auc()
        paddle_auc.update(actual, expected)

        self.assertTrue(np.allclose(result0, paddle_auc.accumulate(), atol=0.0001))


class TestPrecision(unittest.TestCase):
    """
    Test Precision metric
    """
    
    def test_main(self):
        """
        main test case for Precision
        """
        pre = Precision()
        actual = paddle.rand([2, 2])
        expected = paddle.randint(0, 2, [2, 1])

        result0 = pre.compute(actual, expected)
        paddle_pre = paddle.metric.Precision()
        paddle_pre.update(actual[:, 1], expected)

        self.assertTrue(np.allclose(result0, paddle_pre.accumulate(), atol=0.0001))


class TestRecall(unittest.TestCase):
    """
    Test Recall metric
    """
    
    def test_main(self):
        """
        main test case for Recall
        """
        recall = Recall()
        actual = paddle.rand([2, 2])
        expected = paddle.randint(0, 2, [2, 1])

        result0 = recall.compute(actual, expected)
        paddle_rec = paddle.metric.Recall()
        paddle_rec.update(actual[:, 1], expected)

        self.assertTrue(np.allclose(result0, paddle_rec.accumulate(), atol=0.0001))


if __name__ == '__main__':
    unittest.main()
