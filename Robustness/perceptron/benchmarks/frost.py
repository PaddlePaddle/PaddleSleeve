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

"""Metric that tests models against frost variations."""

import numpy as np
from tqdm import tqdm
from collections import Iterable
from .base import Metric
from .base import call_decorator
from PIL import Image
import warnings


class FrostMetric(Metric):
    """Metric that tests models against frost variations."""

    @call_decorator
    def __call__(self, adv, scenario=5, annotation=None, unpack=True,
                 abort_early=True, verify=False, epsilons=1000):
        """Change the frost of the image until it is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray`
            The original, unperturbed input as a `numpy.ndarray`.
        scenario : int or PIL.Image
            Choice of frost backgrounds.
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
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        img_height, img_width = hw

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons)[1:]
        else:
            epsilons = epsilons

        if isinstance(scenario, Image.Image):
            frost_img_pil = scenario
        elif isinstance(scenario, int):
            frost_img_pil = Image.open(
                'perceptron/utils/images/frost{0}.png'.format(scenario))
        else:
            raise ValueError(
                'scenatiro has to be eigher int or PIL.Image.Image')

        frost_img = np.array(
            frost_img_pil.convert('RGB').resize(
                (img_width, img_height))).astype(
            np.float32) / 255.
        frost_img = frost_img * max_
        if (axis == 0):
            frost_img = np.transpose(frost_img, (2, 0, 1))

        cc0 = [1.0, 0.5]
        cc1 = [0.3, 0.8]
        for _, epsilon in enumerate(tqdm(epsilons)):
            p0 = cc0[0] + epsilon * (cc0[1] - cc0[0])
            p1 = cc1[0] + epsilon * (cc1[1] - cc1[0])
            perturbed = image * p0 + frost_img * p1
            perturbed = np.clip(perturbed, min_, max_)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                if abort_early or verify:
                    break
            else:
                bound = epsilon
                a.verifiable_bounds = (bound, None)

        return
