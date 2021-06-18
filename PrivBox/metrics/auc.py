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

"""
AUC metric modulus, used for evaluation
"""


class AUC(Metric):
    """
    AUC metric
    """

    def __init__(self, soft_actual=True):
        """
        init AUC class, only for binary classifier

        Args:
            soft_actual(Boolean): Whether input of actual value is soft label
        """
        self.soft_actual = soft_actual

    def compute(self, actual, expected):
        """
        compute AUC metric

        Args:
            actual(Tensor): Actual result
            expected(Tensor): Expected result
            
        Returns:
            (Tensor): AUC for input of expected and actual
        """

        auc = paddle.metric.Auc()
        if not self.soft_actual:
            shape = actual.shape
            if shape[-1] == 1:
                del shape[-1]
            actual = paddle.nn.functional.one_hot(actual.reshape(shape).astype("int32"), num_classes=2)
        auc.update(actual, expected)
        return auc.accumulate()
