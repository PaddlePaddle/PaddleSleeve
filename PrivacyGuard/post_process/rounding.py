# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
This module provides a rouding defense strategy for inference and extraction attack.
"""

import numpy as np
import paddle
from post_processor import PostProcessor

__all__ = ["RoundingNet"]

class RoundingNet(PostProcessor):
    """
    Rouding strategy round the network output to the defined precision
    """

    def __init__(self,
                 network,
                 precision,
                 use_softmax=True,
                 axes=None,
                 starts=None,
                 ends=None):
        r"""
        Rounding network output to the precision

        Parameters:
            network (Layer): trained paddle network.
            precision (int): the precision for rounding output
            axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to .
            starts (list|tuple|Tensor): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of
                    it should be integers or Tensors with shape [1]. If ``starts`` is an Tensor, it should be an 1-D Tensor.
                    It represents starting indices of corresponding axis in ``axes``.
            ends (list|tuple|Tensor): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                    it should be integers or Tensors with shape [1]. If ``ends`` is an Tensor, it should be an 1-D Tensor .
                    It represents ending indices of corresponding axis in ``axes``.

            Note: Parameters of 'axes', 'starts', 'ends' are used for slice network output, more example see 'paddle.slice' API.
        """

        super(RoundingNet, self).__init__(network, axes, starts, ends)
        self.precision = precision
        self.use_softmax = use_softmax

    def _post_functor(self, x, indices):
        """
        Post processing for network's output,
        return the rounding precision value.

        Parameters:
            x (Tensor): output of network
            indices (tuple[slice]): slice variable of paddle or numpy
        """
        if self.use_softmax:
            x[indices] = paddle.nn.functional.softmax(x[indices])
        y = x.numpy()
        y[indices] = np.round(y[indices], decimals=self.precision)
        return paddle.to_tensor(y)
