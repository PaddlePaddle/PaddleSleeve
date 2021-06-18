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

import paddle
import abc
from .metric import Metric

from paddle.nn.functional import mse_loss
"""
Accuracy metric modulus, used for evaluation
"""


class Accuracy(Metric):
    """
    Accuracy metric
    """

    def __init__(self, soft_actual=True, num_classes=None):
        """
        init Accuracy class

        Args:
            soft_actual(Boolean): Whether input of actual value is soft label
                when set to 'False', must input 'num_classes'
            num_classes(int): number of classes
        """
        self.soft_actual = soft_actual
        self.num_classes = num_classes
        if not soft_actual and num_classes is None:
            raise ValueError("must input num_classes when set soft_actual as False")

    def compute(self, actual, expected):
        """
        compute acc metric

        Args:
            actual(Tensor): Actual result
            expected(Tensor): Expected result
            
        Returns:
            (Tensor): accuracy for input of expected and actual
        """
        if not self.soft_actual:
            shape = actual.shape
            if shape[-1] == 1:
                del shape[-1]
            actual = paddle.nn.functional.one_hot(actual.reshape(shape).astype('int32'), self.num_classes)
        acc_o = paddle.metric.Accuracy()
        correct = acc_o.compute(actual, expected)
        acc_o.update(correct)
        return acc_o.accumulate()
