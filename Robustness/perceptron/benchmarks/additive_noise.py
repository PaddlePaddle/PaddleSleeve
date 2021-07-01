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

"""Metric that tests models against different types of additive noise."""

from abc import abstractmethod
from collections import Iterable
from tqdm import tqdm
import numpy as np
from .base import Metric
from .base import call_decorator
from perceptron.utils.rngs import nprng


class AdditiveNoiseMetric(Metric):
    """Base class for metric that tests models against additive noise."""

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, epsilons=10000):
        """Adds uniform or Gaussian noise to the image, gradually increasing
        the standard deviation until the image is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        annotation : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
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
        bounds = a.bounds()
        min_, max_ = bounds

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in tqdm(epsilons):
            noise = self._sample_noise(epsilon, image, bounds)
            perturbed = image + epsilon * noise
            perturbed = np.clip(perturbed, min_, max_)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial and abort_early:
                return

    @abstractmethod
    def _sample_noise(self):
        raise NotImplementedError


class AdditiveUniformNoiseMetric(AdditiveNoiseMetric):
    """Metric that tests models against uniform noise."""

    def _sample_noise(self, epsilon, image, bounds):
        min_, max_ = bounds
        w = epsilon * (max_ - min_)
        noise = nprng.uniform(-w, w, size=image.shape)
        noise = noise.astype(image.dtype)
        return noise


class AdditiveGaussianNoiseMetric(AdditiveNoiseMetric):
    """Metric that tests models against Gaussian noise."""

    def _sample_noise(self, epsilon, image, bounds):
        min_, max_ = bounds
        std = epsilon / np.sqrt(3) * (max_ - min_)
        noise = nprng.normal(scale=std, size=image.shape)
        noise = noise.astype(image.dtype)
        return noise
