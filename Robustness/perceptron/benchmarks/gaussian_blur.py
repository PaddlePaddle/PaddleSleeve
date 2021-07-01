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

"""Metric that tests models against Gaussian blurs."""

import numpy as np
from tqdm import tqdm
from collections import Iterable
from scipy.ndimage.filters import gaussian_filter
from .base import Metric
from .base import call_decorator


class GaussianBlurMetric(Metric):
    """Metric that tests models against Gaussian blurs."""

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, epsilons=10000):
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
        epsilons : int or Iterable[float]
            Either Iterable of standard deviations of the Gaussian blur
            or number of standard deviations between 0 and 1 that should
            be tried.

        """

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        h, w = hw
        size = max(h, w)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 0.2, num=epsilons + 1)[1:]

        for epsilon in tqdm(epsilons):
            # epsilon = 1 will correspond to
            # sigma = size = max(width, height)
            sigmas = [epsilon * size] * 3
            sigmas[axis] = 0
            blurred = gaussian_filter(image, sigmas)
            blurred = np.clip(blurred, min_, max_)
            _, is_adversarial = a.predictions(blurred)
            if is_adversarial and abort_early:
                return
