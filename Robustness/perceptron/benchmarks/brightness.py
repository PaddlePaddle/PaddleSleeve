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

"""Metric that tests models against brightness variations."""

import numpy as np
from tqdm import tqdm
from collections import Iterable
from .base import Metric
from .base import call_decorator
import warnings


class BrightnessMetric(Metric):
    """Metric that tests models against brightness variations."""

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, verify=False, epsilons=1000):
        """Change the brightness of the image until it is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray`
            The original, unperturbed input as a `numpy.ndarray`.
        annotation : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        abort_early : bool
            If true, returns when got first adversarial, otherwise
            returns when all the iterations are finished.
        verify : bool
            If True, return verifiable bound.
        epsilons : int or Iterable[float]
            Either Iterable of contrast levels or number of brightness
            factors between 1 and 0 that should be tried. Epsilons are
            one minus the brightness factor. Epsilons are not used if
            verify = True.

        """

        if verify is True:
            warnings.warn('epsilon is not used in verification mode '
                          'and abort_early is set to True.')

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()

        if verify:
            step_size = 1 / (255. * 255.)
            epsilons_ub = np.arange(1, 255, step_size)
            epsilons_lb = np.arange(1, 1 / 255., -1 * step_size)
        elif not isinstance(epsilons, Iterable):
            epsilons_ub = np.linspace(1, 255, num=int(epsilons / 2 + 1))[1:]
            epsilons_lb = np.linspace(
                1, 1 / 255., num=int(epsilons - epsilons / 2 + 1))[1:]
        else:
            epsilons_ub = epsilons
            epsilons_lb = []

        epsilon_ub_idx = 0
        epsilon_lb_idx = 0
        upper_bound = 1.
        lower_bound = 1.
        perturbed_ub = np.ones(image.shape)
        perturbed_lb = np.zeros(image.shape)

        for idx, epsilon in enumerate(tqdm(epsilons_ub)):
            perturbed = image * epsilon
            perturbed = np.clip(perturbed, min_, max_)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                epsilon_ub_idx = idx
                perturbed_ub = perturbed
                if abort_early or verify:
                    break
            else:
                upper_bound = epsilon
                a.verifiable_bounds = (upper_bound, lower_bound)

        for idx, epsilon in enumerate(tqdm(epsilons_lb)):
            perturbed = image * epsilon
            perturbed = np.clip(perturbed, min_, max_)
            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                epsilon_lb_idx = idx
                perturbed_lb = perturbed
                lower_bound = epsilon
                if abort_early or verify:
                    break
            else:
                lower_bound = epsilon
                a.verifiable_bounds = (upper_bound, lower_bound)

        return
