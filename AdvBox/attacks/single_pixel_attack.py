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
This module provides the attack method for SinglePixelAttack & LocalSearchAttack's implement.
"""
from __future__ import division

from builtins import zip
from builtins import str
from builtins import range
import logging
from collections import Iterable

logger = logging.getLogger(__name__)

import numpy as np
from .base import Attack
import paddle


__all__ = [
    'SinglePixelAttack'
]


# Simple Black-Box Adversarial Perturbations for Deep Networks
# 随机在图像中选择max_pixels个点 在多个信道中同时进行修改，修改范围通常为0-255
class SinglePixelAttack(Attack):
    """
    SinglePixelAttack
    """

    def __init__(self, model):
        """

        Args:
            model:
            support_targeted:
        """
        super(SinglePixelAttack, self).__init__(model)

    # 如果输入的原始数据，isPreprocessed为False，如果驶入的图像数据被归一化了，设置为True
    def _apply(self, adversary, max_pixels=1000):
        """

        Args:
            adversary:
            max_pixels:

        Returns:

        """
        if adversary.is_targeted_attack:
            raise ValueError(
                "This attack method doesn't support targeted attack!")

        min_, max_ = self.model.bounds

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.denormalized_original)

        axes = [i for i in range(adversary.original.ndim) if i != self.model.input_channel_axis]

        # 输入的图像必须具有长和宽属性
        assert len(axes) == 2

        h = adv_img.shape[axes[0]]
        w = adv_img.shape[axes[1]]

        # print("w={0},h={1}".format(w,h))
        # max_pixels为攻击点的最多个数 从原始图像中随机选择max_pixels个进行攻击

        pixels = np.random.permutation(h * w)
        pixels = pixels[:max_pixels]

        for i, pixel in enumerate(pixels):
            x = pixel % w
            y = pixel // w

            location = [x, y]
            if i % 50 == 0:
                logging.info("Attack location x={0} y={1}".format(x, y))

            location.insert(self.model.input_channel_axis, slice(None))
            location = tuple(location)

            # TODO: add differential evolution
            for value in [min_, max_]:
                perturbed = np.copy(adv_img)
                # 针对图像的每个信道的点[x,y]同时进行修改
                perturbed[location] = value
                perturbed = paddle.to_tensor(perturbed, dtype='float32', place=self._device)

                perturbed_normalized = self.input_preprocess(perturbed)
                adv_label = np.argmax(self.model.predict(perturbed_normalized))

                perturbed = self.safe_delete_batchsize_dimension(perturbed)
                perturbed_normalized = self.safe_delete_batchsize_dimension(perturbed_normalized)
                is_ok = adversary.try_accept_the_example(perturbed.numpy(),
                                                         perturbed_normalized.numpy(),
                                                         adv_label)
                if is_ok:
                    return adversary

        return adversary
