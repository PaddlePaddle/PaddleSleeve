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
import numpy as np

from paddle.nn.functional import mse_loss

import paddle.nn.functional as F

from .metric import Metric
"""
Structural similarity (SSIM) metrics modulus, used for evaluation
"""

class SSIM(Metric):
    """
    Structural similarity metric
    """

    def __init__(self, channel=1, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        self.channel = channel

    def compute(self, actual, expected):
        """
        compute structural similarity

        Args:
            real(Tensor): Actual image
            expected(Tensor): Expected image
        
        Returns:
            (float): Mean square error for input of expected and actual
        """
     
        return self._ssim(actual, expected, self.filter_size,
                          self.filter_sigma, self.channel, self.max_val,
                          self.k1, self.k2)[0]

    def _ssim(self, img1, img2, filter_size, filter_sigma,
              channel=3, max_val=1.0, k1=0.01, k2=0.03):
        """
        Computes SSIM index between img1 and img2,
        based on the standard SSIM implementation from:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
        quality assessment: from error visibility to structural similarity. IEEE
        transactions on image processing.
        """
        C1 = (k1 * max_val) ** 2
        C2 = (k2 * max_val) ** 2

        window = self._create_window(filter_size, filter_sigma, channel)
        padding = filter_size // 2

        mu_x = F.conv2d(img1, window, padding=padding, groups=channel)
        mu_y = F.conv2d(img2, window, padding=padding, groups=channel)
        
        mu_x_sqr = mu_x.square()
        mu_y_sqr = mu_y.square()
        mu_xy = mu_x * mu_y
        sigma_x_sqr = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu_x_sqr
        sigma_y_sqr = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu_y_sqr
        sigma_xy = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu_xy
        
        # calc SSIM constrast-structure
        cs = (2 * sigma_xy + C2) / (sigma_x_sqr + sigma_y_sqr + C2)
        # calc SSIM luminance
        luminance = (2 * mu_xy + C1) / (mu_x_sqr + mu_y_sqr + C1)

        ssim = (cs * luminance).mean()
        return ssim.numpy()

    def _gauss_kernel(self, filter_size, filter_sigma):
        x = paddle.arange(filter_size, dtype='float32')
        x -= filter_size // 2

        g = -x ** 2 / float(2 * filter_sigma ** 2)
        soft_g = F.softmax(g)

        return soft_g

    def _create_window(self, filter_size, filter_sigma, channel):
        _1D_window = self._gauss_kernel(filter_size, filter_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window.expand([channel, 1, filter_size, filter_size])

