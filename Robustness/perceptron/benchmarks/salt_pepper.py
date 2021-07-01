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

"""Metric that tests models against salt and pepper noise."""

import numpy as np
from tqdm import tqdm
from collections import Iterable
from .base import Metric
from .base import call_decorator
from perceptron.utils.rngs import nprng


class SaltAndPepperNoiseMetric(Metric):
    """Add salt and pepper noise."""

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, epsilons=10000, repetitions=10):
        """Add salt and pepper noise.

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
            Either Iterable of standard deviations of the salt and pepper
            or number of standard deviations between 0 and 1 that should
            be tried.
        repetitions : int
            Specifies how often the attack will be repeated.

        """

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        channels = image.shape[axis]
        shape = list(image.shape)
        shape[axis] = 1
        r = max_ - min_
        pixels = np.prod(shape)

        max_epsilon = 1
        is_preset_eps = False
        if not isinstance(epsilons, Iterable):
            epsilon_n_steps = min(epsilons, pixels)
        else:
            is_preset_eps = True

        for _ in tqdm(range(repetitions)):
            if not is_preset_eps:
                epsilons = np.linspace(
                    0, max_epsilon, num=epsilon_n_steps + 1)[1:]
            for epsilon in epsilons:
                p = epsilon

                u = nprng.uniform(size=shape)
                u = u.repeat(channels, axis=axis)

                salt = (u >= 1 - p / 2).astype(image.dtype) * r
                pepper = -(u < p / 2).astype(image.dtype) * r

                perturbed = image + salt + pepper
                perturbed = np.clip(perturbed, min_, max_)

                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = a.predictions(perturbed)
                if is_adversarial:
                    # higher epsilon usually means larger perturbation, but
                    # this relationship is not strictly monotonic, so we set
                    # the new limit a bit higher than the best one so far
                    if abort_early:
                        return
                    max_epsilon = epsilon * 1.2
                    break
