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

"""Metric that tests models against snow variations."""

import numpy as np
from tqdm import tqdm
from collections import Iterable
from .base import Metric
from .base import call_decorator
from PIL import Image
import warnings
from perceptron.benchmarks.motion_blur import MotionBlurMetric


class SnowMetric(Metric):
    """Metric that tests models against snow variations."""

    @call_decorator
    def __call__(self, adv, angle=45, annotation=None, unpack=True,
                 abort_early=True, verify=False, epsilons=1000):
        """Change the snow of the image until it is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray`
            The original, unperturbed input as a `numpy.ndarray`.
        angle : float
            Angle of snowfall.
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
        import cv2

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

        snow_mask_np = np.zeros((img_height // 10, img_height // 10, 3))
        ch = snow_mask_np.shape[0] // 2
        cw = snow_mask_np.shape[1] // 2
        cr = min(img_height, img_width) * 0.1
        for i in range(snow_mask_np.shape[0]):
            for j in range(snow_mask_np.shape[1]):
                if (i - ch) ** 2 + (j - cw) ** 2 <= cr:
                    snow_mask_np[i, j] = np.ones(3)

        kernel = MotionBlurMetric.motion_Kernel((int(ch * 0.9),
                                                 int(cw * 0.9)),
                                                angle)
        blured = cv2.filter2D(snow_mask_np, -1, kernel)
        blured = np.clip(blured, min_, max_).astype(np.float32)
        blured = blured * max_
        blured_h, blured_w = blured.shape[:2]
        if axis == 0:
            blured = np.transpose(blured, (2, 0, 1))

        cc0 = [1, 100]
        for _, epsilon in enumerate(tqdm(epsilons)):
            p0 = int(cc0[0] + epsilon * (cc0[1] - cc0[0]))
            positions_h = np.random.randint(img_height - blured_h, size=p0)
            positions_w = np.random.randint(img_width - blured_w, size=p0)
            perturbed = np.copy(image)
            for temp_h, temp_w in zip(positions_h, positions_w):
                if axis == 0:
                    perturbed[:, temp_h: temp_h + blured_h, temp_w: temp_w + blured_w] += blured
                else:
                    perturbed[temp_h: temp_h + blured_h, temp_w: temp_w + blured_w, :] += blured
            perturbed = np.clip(perturbed, min_, max_)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                if abort_early or verify:
                    break
            else:
                bound = epsilon
                a.verifiable_bounds = (bound, None)

        return
