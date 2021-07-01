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

"""Metric that tests models against translations."""

from abc import abstractmethod
import numpy as np
from tqdm import tqdm
from collections import Iterable
import math
from .base import Metric
from .base import call_decorator
import warnings


class TranslationMetric(Metric):
    """Metric that tests models against translations."""

    @call_decorator
    def __call__(self, adv, pix_range=None, annotation=None, unpack=True,
                 abort_early=True, verify=False, epsilons=100):
        """Translate the image until it is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray`
            The original, unperturbed input as a `numpy.ndarray`.
        pix_range : int or (int, int)
            pix_range of pixels for translation attack
        annotation : int
            The reference label of the original input.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        abort_early : bool
            If true, returns when got first adversarial, otherwise
            returns when all the iterations are finished.
        verify : bool
            if True, return verifiable bound
        epsilons : int or Iterable[float]
            Either Iterable of translation distances or number of
            translation levels between 1 and 0 that should be tried.
            Epsilons are one minus the contrast level. Epsilons are
            not used if verify = True.

        """

        if verify is True:
            warnings.warn('epsilon is not used in verification mode '
                          'and abort_early is set to True.')

        if isinstance(pix_range, int):
            pix_range = (pix_range, pix_range)
        if pix_range:
            assert len(
                pix_range) == 2, "pix_range has to be float of pix_range or' \
                    ' (pix_range_low, pix_range_high)"
            assert pix_range[0] <= pix_range[1], "pix_range[0] should be'\
                ' smaller than pix_range[1]"

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        img_width, img_height = hw
        translate_type, translate_max_bound = self._get_type(hw)
        if verify:
            epsilons_ub = np.arange(1, translate_max_bound, 1)
            epsilons_lb = np.arange(-1, -1 * translate_max_bound, -1)
        elif not isinstance(epsilons, Iterable):
            if not pix_range:
                range_max = translate_max_bound
                range_min = -1 * translate_max_bound
            else:
                range_max = pix_range[1]
                range_min = pix_range[0]

            if range_min >= 0:
                epsilons = np.minimum(translate_max_bound, epsilons)
                epsilons_ub = np.linspace(
                    range_min, range_max, num=epsilons)
                epsilons_lb = []
            elif range_max <= 0:
                epsilons = np.minimum(translate_max_bound, epsilons)
                epsilons_ub = []
                epsilons_lb = np.linspace(
                    range_max, range_min, num=epsilons)
            else:
                epsilons = np.minimum(2 * translate_max_bound, epsilons)
                epsilons_ub = np.linspace(
                    0, range_max, num=int(epsilons / 2 + 1))[1:]
                epsilons_lb = np.linspace(
                    0, range_min, num=int(epsilons / 2 + 1))[1:]
        else:
            epsilons_ub = epsilons
            epsilons_lb = []

        epsilons_ub = epsilons_ub.astype(int)
        epsilons_lb = epsilons_lb.astype(int)
        upper_bound = 0
        lower_bound = 0

        if axis == 0:
            image_cv = np.transpose(image, (1, 2, 0))
        elif axis == 2:
            image_cv = np.copy(image)
        else:
            raise ValueError('Invalid axis.')

        import cv2

        print('Generating adversarial examples.')
        for idx, epsilon in enumerate(tqdm(epsilons_ub)):
            epsilon = int(epsilon)
            if translate_type == 'horizontal':
                M = np.float32([[1, 0, epsilon], [0, 1, 0]])
            elif translate_type == 'vertical':
                M = np.float32([[1, 0, 0], [0, 1, epsilon]])
            else:
                raise ValueError('Invalid translate_type')

            perturbed = cv2.warpAffine(image_cv, M, (img_width, img_height))
            if axis == 0:
                perturbed = np.transpose(perturbed, (2, 0, 1))

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
            epsilon = int(epsilon)
            if translate_type == 'horizontal':
                M = np.float32([[1, 0, epsilon], [0, 1, 0]])
            elif translate_type == 'vertical':
                M = np.float32([[1, 0, 0], [0, 1, epsilon]])
            else:
                raise ValueError('Invalid translate_type')
            perturbed = cv2.warpAffine(image_cv, M, (img_width, img_height))
            if axis == 0:
                perturbed = np.transpose(perturbed, (2, 0, 1))

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                epsilon_lb_idx = idx
                perturbed_lb = perturbed
                if abort_early or verify:
                    break
            else:
                lower_bound = epsilon
                a.verifiable_bounds = (upper_bound, lower_bound)

        return

    @abstractmethod
    def _get_type(self, hw):
        raise NotImplementedError


class HorizontalTranslationMetric(TranslationMetric):
    """Horizontally Translate the image until it is misclassified."""

    def _get_type(self, hw):
        return 'horizontal', hw[1]


class VerticalTranslationMetric(TranslationMetric):
    """Vertically Translate the image until it is misclassified."""

    def _get_type(self, hw):
        return 'vertical', hw[0]
