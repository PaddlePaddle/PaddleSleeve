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

import privbox.attack
import paddle
import numpy as np
from paddle import nn

from privbox.metrics import MSE, Accuracy, AUC, Recall, Precision, PSNR, SSIM


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


class TestPSNR(unittest.TestCase):
    """
    Test PSNR metric
    """
    
    def test_main(self):
        """
        main test case for psnr
        """
        psnr = PSNR()
        image0 = paddle.rand([1, 1, 64, 64])
        image1 = image0.clone()
        image1[0, 0, 1, 1] = image0[0, 0, 1, 1] + 0.00001
        
        # two similar images have high PSNR
        result = psnr.compute(image0, image1)
        self.assertTrue(result > 50)

        result1 = psnr.compute(image0, image0)
        self.assertTrue(result1 > 50)
        
        # two random images have low PSNR
        image2 = paddle.rand([1, 1, 64, 64])
        result2 = psnr.compute(image0, image2)
        self.assertTrue(result2 < 10)

    def test_channel3(self):
        """
        psnr test case for 3 channel image
        """
        psnr = PSNR()
        image0 = paddle.rand([1, 3, 64, 64])
        image1 = image0.clone()
        image1[0, 0, 1, 1] = image0[0, 0, 1, 1] + 0.00001

        # two similar images have high PSNR
        result = psnr.compute(image0, image1)
        self.assertTrue(result > 50)

        result1 = psnr.compute(image0, image0)
        self.assertTrue(result1 > 50)

        # two random images have low PSNR
        image2 = paddle.rand([1, 3, 64, 64])
        result2 = psnr.compute(image0, image2)
        self.assertTrue(result2 < 10)


class TestSSIM(unittest.TestCase):
    """
    Test SSIM metric
    """
    
    def test_main(self):
        """
        main test case for ssim
        """
        ssim = SSIM()
        image0 = paddle.rand([1, 1, 64, 64])
        image1 = image0.clone()
        image1[0, 0, 1, 1] = image0[0, 0, 1, 1] + 0.00001
        # two similar images have high SSIM
        result = ssim.compute(image0, image1)
        self.assertTrue(result > 0.9)

        result1 = ssim.compute(image0, image0)
        self.assertTrue(result1 > 0.9)
        
        # two random images have low PSNR
        image2 = paddle.rand([1, 1, 64, 64])
        result2 = ssim.compute(image0, image2)
        self.assertTrue(result2 < 0.1)

    def test_channel3(self):
        """
        ssim test case for 3 channel image
        """
        ssim = SSIM(channel=3)
        image0 = paddle.rand([1, 3, 64, 64])
        image1 = image0.clone()
        image1[0, 0, 1, 1] = image0[0, 0, 1, 1] + 0.00001
        # two similar images have high SSIM
        result = ssim.compute(image0, image1)
        self.assertTrue(result > 0.9)

        result1 = ssim.compute(image0, image0)
        self.assertTrue(result1 > 0.9)
        
        # two random images have low PSNR
        image2 = paddle.rand([1, 3, 64, 64])
        result2 = ssim.compute(image0, image2)
        self.assertTrue(result2 < 0.1)


if __name__ == '__main__':
    unittest.main()
