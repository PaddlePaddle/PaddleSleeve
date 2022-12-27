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
import math

from paddle.nn.functional import mse_loss

from .metric import Metric
"""
Peak signal-to-noise ratio (PSNR) metrics modulus, used for evaluation
"""

class PSNR(Metric):
    """
    Peak signal-to-noise ratio metric
    """

    def __init__(self, max_val=1.0):
        self.max_val = max_val

    def compute(self, actual, expected):
        """
        compute peak signal-to-noise ratio

        Args:
            real(Tensor): Actual result
            expected(Tensor): Expected result
        
        Returns:
            (float): Mean square error for input of expected and actual
        """
        mse = float(mse_loss(actual, expected, reduction="mean"))

        if mse == 0:
            # prevent divided by zero
            mse = 0.000001
        psnr = 20 * math.log10(self.max_val / math.sqrt(mse))
        return psnr
