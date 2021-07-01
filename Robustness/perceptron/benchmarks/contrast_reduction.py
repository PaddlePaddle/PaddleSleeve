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

"""Metric that tests models against contrast reductions."""

import numpy as np
from collections import Iterable
from tqdm import tqdm
from .base import Metric
from .base import call_decorator


class ContrastReductionMetric(Metric):
    """Metric that tests models against brightness variations."""

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True, abort_early=True,
                 threshold=1.0, epsilons=1000):
        """Reduces the contrast of the image until it is misclassified.

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
        threshold : float
            Upper bound for contrast factor
        epsilons : int or Iterable[float]
            Either Iterable of contrast levels or number of contrast
            levels between 1 and 0 that should be tried. Epsilons are
            one minus the contrast level.

        """

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        target = (max_ + min_) / 2

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, threshold, num=epsilons + 1)[1:]

        for epsilon in tqdm(epsilons):
            perturbed = (1 - epsilon) * image + epsilon * target

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial and abort_early:
                return
