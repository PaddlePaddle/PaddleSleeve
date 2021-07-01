# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""Metric that tests models against motion blurs."""
from .base import call_decorator
from .base import Metric
from scipy.ndimage.filters import gaussian_filter
import math
from collections import Iterable
from tqdm import tqdm
import numpy as np


class MotionBlurMetric(Metric):
    """Motion blurs the image until it is misclassified."""

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, motion_angle=0, epsilons=10000):
        """Blurs the image until it is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray`
            The original, unperturbed input as a `numpy.ndarray`.
        annotation : int
            The reference label of the original input.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        abort_early : bool
            If true, returns when got first adversarial, otherwise
            returns when all the iterations are finished.
        motion_angle : float
            Motion angle in degree between 0 and 180.
        epsilons : int or Iterable[float]
            Either Iterable of kernel size that should
            be tried.

        """
        import cv2

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        h, w = hw
        size_min = min(h, w)

        if axis == 0:
            image_cv = np.transpose(image, (1, 2, 0))
        elif axis == 2:
            image_cv = np.copy(image)
        else:
            raise ValueError('Invalid axis.')

        if epsilons > size_min:
            epsilons = size_min
        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, size_min, num=epsilons + 1)[1:]
        for epsilon in tqdm(epsilons):
            kernel = MotionBlurMetric.motion_Kernel((epsilon, epsilon), motion_angle)
            blurred = cv2.filter2D(image_cv, -1, kernel)
            blurred = np.clip(blurred, min_, max_)
            if axis == 0:
                blurred = np.transpose(blurred, (2, 0, 1))
            _, is_adversarial = a.predictions(blurred)
            if is_adversarial and abort_early:
                return

    @staticmethod
    def motion_Kernel(dim, angle):

        if isinstance(dim, int):
            dim = (dim, dim)
        dim = np.array(dim).astype(int)
        center = (dim - 1) / 2
        kernel = np.zeros(dim)
        angle = math.fmod(angle, 180.0)
        if angle == 90:
            kernel[:, int(center[1])] = 1
            return kernel

        delta_x = 0
        delta_y = 0
        while True:
            if (center[1] + delta_x).is_integer():
                if int(round(center[0] + delta_y)) < 0 or int(round(center[0] + delta_y)) >= dim[0] or int(
                        center[1] + delta_x) < 0 or int(center[1] + delta_x) >= dim[1]:
                    break

                kernel[int(round(center[0] - delta_y)),
                       int(center[1] + delta_x)] = 1
                kernel[int(round(center[0] + delta_y)),
                       int(center[1] - delta_x)] = 1
            delta_x += 0.5
            delta_y = delta_x * math.tan(angle / 180 * np.pi)
        normalizationFactor = np.count_nonzero(kernel)
        kernel /= normalizationFactor
        return kernel
