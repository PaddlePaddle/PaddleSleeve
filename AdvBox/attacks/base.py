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
The base model of the model.
"""
import logging
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import paddle


class Attack(object):
    """
    Abstract base class for adversarial attacks. `Attack` represent an
    adversarial attack which search an adversarial example. Subclass should
    implement the _apply(self, adversary, **kwargs) method.
    Args:
        model(Model): an instance of a paddle model
    """
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model
        self._device = paddle.get_device()

    def __call__(self, adversary, **kwargs):
        """
        Generate the adversarial sample.
        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        # TODO: add epsilon-ball transform computation. epsilon= 2/255, 8/255 => transformed_epsilon
        # TODO: make user specify normalization setting.
        return self._apply(adversary, **kwargs)

    @abstractmethod
    def _apply(self, adversary, **kwargs):
        """
        Search an adversarial example.
        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        raise NotImplementedError
