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


class AccMetric(Metric):
    """
    Accuracy metric
    """

    def compute(self, expected, real):
        """
        compute acc metric

        Args:
            expected(Tensor): Expected result
            real(Tensor): Actual result
        
        Returns:
            (Tensor): accuracy for input of expected and real
        """
        acc_o = paddle.metric.Accuracy()
        return acc_o.compute(expected, real)
