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
This module provides a base class for defense strategy of post processing method.
"""

import numpy as np
import paddle

__all__ = ["PostProcessor"]

class PostProcessor(paddle.nn.Layer):
    """
    Post-Processing defense strategy, 
    uses non-intrusive method to enhance the security of the model, 
    and reduces the privacy disclosure caused by the model output.

    Notice: Post-processing models cannot be used directly for training, 
    and if training is required, 
    the original network model needs to be obtained through the 'origin_network()' API for training.
    """

    def __init__(self, network, axes=None, starts=None, ends=None):
        r"""
        Init network

        Parameters:

        network (Layer): A trained paddle network.

        axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to .
        starts (list|tuple|Tensor): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``starts`` is an Tensor, it should be an 1-D Tensor.
                It represents starting indices of corresponding axis in ``axes``.
        ends (list|tuple|Tensor): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``ends`` is an Tensor, it should be an 1-D Tensor .
                It represents ending indices of corresponding axis in ``axes``.

        Note: Parameters of 'axes', 'starts', 'ends' are used for slice network output, more example see 'paddle.slice' API.
        """

        super(PostProcessor, self).__init__()
        if not isinstance(network, paddle.nn.Layer):
            ValueError("input network type must be 'paddle.nn.Layer'")
        self.net = network
        self.axes = axes
        self.starts = starts
        self.ends = ends

    def forward(self, x):
        """
        Forward computing
        """
        y = self.net(x)
        indices = self._get_indice(y)
        y = self._post_functor(y, indices)
        return y

    def backward(self, *inputs):
        """
        PostProcessor is only used for inference,
        Train PostProcessor is not allowed.
        However, you can get origin network by `origin_network()` api for training
        """
        raise ValueError("""
                PostProcessor is only used for inference, 
                Training a PostProcessor is not allowed.
                However, you can get origin network by `origin_network()` api for training.""")

    def origin_network(self):
        """
        return origin network
        """
        return self.net

    def _post_functor(self, x, indices):
        """
        Post processing functor, must be overrided.

        Parameters:
            x (Tensor): output of origin network
            indices (tuple[slice]): slice variable of paddle or numpy
        """
        ValueError("'_post_functor() method must be overrided in sub-class'")
        return x

    def _get_indice(self, x):
        indices = []
        for i in range(x.ndim):
            if self.axes is not None and\
                self.starts is not None and\
                self.ends is not None and\
                i in self.axes:
                idx = self.axes.index(i)
                indices.append(slice(self.starts[idx], self.ends[idx], None))
            else:
                indices.append(slice(None, None, None))

        return tuple(indices)