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

"""Metric that tests models against blended uniform noise."""

import logging
import warnings
from collections import Iterable
import numpy as np
from tqdm import tqdm
from .base import Metric
from .base import call_decorator
from perceptron.utils.rngs import nprng


class BlendedUniformNoiseMetric(Metric):
    """Blends the image with a uniform noise image until it
    is misclassified.
    """

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, epsilons=10000, max_directions=1000):
        """Metric that tests models against blended uniform noise.

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
            Either Iterable of standard deviations of the blended noise
            or number of standard deviations between 0 and 1 that should
            be tried.
        max_directions : int
            Maximum number of random images to try.

        """

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()

        for j in tqdm(range(max_directions)):
            # random noise images tend to be classified into the same class,
            # so we might need to make very many draws if the original class
            # is that one
            random_image = nprng.uniform(
                min_, max_, size=image.shape).astype(image.dtype)
            _, is_adversarial = a.predictions(random_image)
            if is_adversarial:
                logging.info('Found adversarial image after {} '
                             'attempts'.format(j + 1))
                break
        else:
            # never breaked
            warnings.warn('BlendedUniformNoiseAttack failed to draw a'
                          ' random image that is adversarial.')

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        for epsilon in tqdm(epsilons):
            perturbed = (1 - epsilon) * image + epsilon * random_image
            # due to limited floating point precision,
            # clipping can be required
            if not a.in_bounds(perturbed):  # pragma: no cover
                np.clip(perturbed, min_, max_, out=perturbed)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial and abort_early:
                return
