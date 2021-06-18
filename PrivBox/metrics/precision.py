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
Precision metric modulus, used for evaluation
"""


class Precision(Metric):
    """
    Precision metric
    """

    def compute(self, actual, expected):
        """
        compute precision metric

        Args:
            actual(Tensor): Actual result
            expected(Tensor): Expected result
            
        Returns:
            (float): Precision for input of expected and actual
        """
        if len(actual.shape) > 1 and actual.shape[-1] == 2:
            actual = actual[:, 1]
        elif len(actual.shape) > 1 and actual.shape[-1] > 2:
            raise ValueError("""Input actual error,
                             Precision metric only supports binary classification.""")

        pre = paddle.metric.Precision()
        pre.update(actual, expected)
        return pre.accumulate()
