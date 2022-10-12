# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Operators for depreprocess sample transformation.
"""

import uuid
import numpy as np
import cv2
from abc import abstractmethod

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Integral


class BaseDePreOperators(object):
    """

    Args:
        name: str. name for an depreprocess operator.
    """
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        # TODO: check name.
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, preprocessed_sample, **kwargs):
        # TODO: add assert preprocessed_sample format as CHW
        depreprocessed_sample = self._apply(preprocessed_sample)

        return depreprocessed_sample

    @abstractmethod
    def _apply(self, preprocessed_sample, **kwargs):
        """
        Transform a preprocessed input back to original value format.
        Args:
        preprocessed_sample: numpy.ndarray. The preprocessed input sample.
        **kwargs: Other named arguments.

        return:
            numpy.ndarray. input sample before preprocessing.
        """
        raise NotImplementedError


class Encode(BaseDePreOperators):
    """
    Detransformation for Decode.
    """
    def __init__(self, name=None):
        super(Encode, self).__init__(name)

    def _apply(self, preprocessed_sample, **kwargs):
        depreprocessed_sample = cv2.cvtColor(preprocessed_sample, cv2.COLOR_RGB2BGR)

        return depreprocessed_sample


class Resize(BaseDePreOperators):
    """
    Detransformation for Resize.
    """
    def __init__(self, target_size, keep_ratio, interp, name=None):
        super(Resize, self).__init__(name)
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.resize_h = target_size[0]
        self.resize_w = target_size[1]

    def apply_image(self, image):
        image = image.astype('uint8')
        return cv2.resize(image, (self.resize_w, self.resize_h), interpolation=self.interp)

    def _apply(self, preprocessed_sample, **kwargs):
        depreprocessed_sample = self.apply_image(preprocessed_sample)

        return depreprocessed_sample


class DenormalizeImage(BaseDePreOperators):
    """
    Detransformation for Normalization.
    """
    def __init__(self, mean, std, input_channel_axis, is_scale=True, name=None):
        super(DenormalizeImage, self).__init__(name)
        self.mean = mean
        self.std = std
        self.input_channel_axis = input_channel_axis
        self.is_scale = is_scale

    def _apply(self, preprocessed_sample, **kwargs):
        assert preprocessed_sample.shape[self.input_channel_axis] == len(self.mean)

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        depreprocessed_sample = preprocessed_sample.numpy()
        depreprocessed_sample *= std
        depreprocessed_sample += mean

        if self.is_scale:
            depreprocessed_sample = depreprocessed_sample * 255.0

        return depreprocessed_sample


class ImPermute(BaseDePreOperators):
    """
    Change the channel from (C, H, W) back to be (H, W, C)
    """
    def __init__(self, name=None):
        super(ImPermute, self).__init__(name)

    def _apply(self, preprocessed_sample, **kwargs):
        depreprocessed_sample = preprocessed_sample.transpose((1, 2, 0))

        return depreprocessed_sample
