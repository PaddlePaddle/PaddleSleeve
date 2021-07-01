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

"""Metric that tests models against rotations."""

import numpy as np
from tqdm import tqdm
from collections import Iterable

import math
from .base import Metric
from .base import call_decorator
import warnings


class RotationMetric(Metric):
    """Metric that tests models against rotations."""

    @call_decorator
    def __call__(self, adv, ang_range=None, annotation=None, unpack=True,
                 abort_early=True, verify=False, epsilons=1000):
        """Rotate the image until it is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray`
            The original, unperturbed input as a `numpy.ndarray`.
        ang_range : (float, float)
            Range of angles for rotation metric.
        annotation : int
            The reference label of the original input.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        abort_early : bool
            If true, returns when got first adversarial, otherwise
            returns when all the iterations are finished.
        verify : bool
            if True, return verifiable bound.
        epsilons : int or Iterable[float]
            Either Iterable of rotation angles or number of angles
            between 1 and 0 that should be tried. Epsilons are
            one minus the contrast level. Epsilons are not used if
            verify = True.

        """

        if verify is True:
            warnings.warn('epsilon is not used in verification mode '
                          'and abort_early is set to True.')

        if ang_range:
            assert len(
                ang_range) == 2, "ang_range has to be (range_low, range_high)"
            assert ang_range[0] <= ang_range[1], "ang_range[0] should be smaller than ang_range[1]"

        a = adv
        del adv
        del annotation
        del unpack

        image = a.original_image
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        img_height, img_width = hw

        if verify:
            epsilons_ub, epsilons_lb = self._get_verify_angle(
                image.shape[1:], ang_range=ang_range)
        elif not isinstance(epsilons, Iterable):
            if not ang_range:
                ang_max = np.pi
                ang_min = -1 * np.pi
            else:
                ang_max = ang_range[1]
                ang_min = ang_range[0]

            if ang_min >= 0:
                epsilons_ub = np.linspace(
                    ang_min, ang_max, num=epsilons + 1)[1:]
                epsilons_lb = []
            elif ang_max <= 0:
                epsilons_ub = []
                epsilons_lb = np.linspace(
                    ang_max, ang_min, num=epsilons + 1)[1:]
            else:
                epsilons_ub = np.linspace(0, ang_max, num=int(epsilons / 2 + 1))[1:]
                epsilons_lb = np.linspace(0, ang_min, num=int(epsilons / 2 + 1))[1:]
        else:
            epsilons_ub = epsilons
            epsilons_lb = []

        epsilon_ub_idx = 0
        epsilon_lb_idx = 0
        upper_bound = 0.
        lower_bound = 0.
        perturbed_ub = image
        perturbed_lb = image

        if axis == 0:
            image_cv = np.transpose(image, (1, 2, 0))
        elif axis == 2:
            image_cv = np.copy(image)
        else:
            raise ValueError('Invalid axis.')
        
        #lazy import
        import cv2
        print('Generating adversarial examples.')
        for idx, epsilon in enumerate(tqdm(epsilons_ub)):
            M = cv2.getRotationMatrix2D(
                (img_width / 2., img_height / 2.), epsilon * 180 / np.pi, 1)
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
            M = cv2.getRotationMatrix2D(
                (img_width / 2., img_height / 2.), epsilon * 180 / np.pi, 1)
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

    def _get_crit_angles(self, img_height, img_width, ang_range=None):
        angles_set = set()
        if not ang_range:
            ang_max = 2 * np.pi
            ang_min = -2 * np.pi
        else:
            ang_max = ang_range[1]
            ang_min = ang_range[0]

        ori_x = (img_width - 1) / 2.
        ori_y = (img_height - 1) / 2.
        print('Calculating critic angles.')
        for i in tqdm(range(img_height)):
            for j in range(img_width):
                pt_x = j - ori_x
                pt_y = -1 * (i - ori_y)
                v1 = [pt_x, pt_y]
                dis = (pt_x ** 2 + pt_y ** 2) ** 0.5
                if dis == 0:
                    continue

                offset = 0.5
                while True:
                    c_x = pt_x + offset
                    if pt_x ** 2 + pt_y ** 2 < c_x ** 2:
                        break
                    elif pt_x ** 2 + pt_y ** 2 == c_x ** 2:
                        c_y = 0
                        v2 = [c_x, c_y]
                        temp_angle = self._get_angle(v1, v2)
                        if temp_angle < ang_min or temp_angle > ang_max:
                            break
                        if temp_angle not in angles_set:
                            angles_set.add(temp_angle)
                            if -1 * temp_angle >= ang_min and -1 * temp_angle <= ang_max:
                                angles_set.add(-1 * temp_angle)
                    else:
                        c_y = (pt_x ** 2 + pt_y ** 2 - c_x ** 2) ** 0.5
                        v2_1 = [c_x, c_y]
                        temp_angle_1 = self._get_angle(v1, v2_1)
                        v2_2 = [c_x, -1 * c_y]
                        temp_angle_2 = self._get_angle(v1, v2_2)
                        if temp_angle_1 >= ang_min and temp_angle_1 <= ang_max:
                            if temp_angle_1 not in angles_set:
                                angles_set.add(temp_angle_1)
                                if -1 * temp_angle_1 >= ang_min and -1 * temp_angle_1 <= ang_max:
                                    angles_set.add(-1 * temp_angle_1)
                        elif temp_angle_2 >= ang_min and temp_angle_2 <= ang_max:
                            if temp_angle_2 not in angles_set:
                                angles_set.add(temp_angle_2)
                                if -1 * temp_angle_2 >= ang_min and -1 * temp_angle_2 <= ang_max:
                                    angles_set.add(-1 * temp_angle_2)
                        else:
                            break
                    offset += 1

                offset = 0.5
                while True:
                    c_x = pt_x - offset
                    if pt_x ** 2 + pt_y ** 2 < c_x ** 2:
                        break
                    elif pt_x ** 2 + pt_y ** 2 == c_x ** 2:
                        c_y = 0
                        v2 = [c_x, c_y]
                        temp_angle = self._get_angle(v1, v2)
                        if temp_angle < ang_min or temp_angle > ang_max:
                            break
                        if temp_angle not in angles_set:
                            angles_set.add(temp_angle)
                            if -1 * temp_angle >= ang_min and -1 * temp_angle <= ang_max:
                                angles_set.add(-1 * temp_angle)
                    else:
                        c_y = (pt_x ** 2 + pt_y ** 2 - c_x ** 2) ** 0.5
                        v2_1 = [c_x, c_y]
                        temp_angle_1 = self._get_angle(v1, v2_1)
                        v2_2 = [c_x, -1 * c_y]
                        temp_angle_2 = self._get_angle(v1, v2_2)
                        if temp_angle_1 >= ang_min and temp_angle_1 <= ang_max:
                            if temp_angle_1 not in angles_set:
                                angles_set.add(temp_angle_1)
                                if -1 * temp_angle_1 >= ang_min and -1 * temp_angle_1 <= ang_max:
                                    angles_set.add(-1 * temp_angle_1)
                        elif temp_angle_2 >= ang_min and temp_angle_2 <= ang_max:
                            if temp_angle_2 not in angles_set:
                                angles_set.add(temp_angle_2)
                                if -1 * temp_angle_2 >= ang_min and -1 * temp_angle_2 <= ang_max:
                                    angles_set.add(-1 * temp_angle_2)
                        else:
                            break
                    offset += 1

                offset = 0.5
                while True:
                    c_y = pt_y + offset
                    if pt_x ** 2 + pt_y ** 2 < c_y ** 2:
                        break
                    elif pt_x ** 2 + pt_y ** 2 == c_y ** 2:
                        c_x = 0
                        v2 = [c_x, c_y]
                        temp_angle = self._get_angle(v1, v2)
                        if temp_angle < ang_min or temp_angle > ang_max:
                            break
                        if temp_angle not in angles_set:
                            angles_set.add(temp_angle)
                            if -1 * temp_angle >= ang_min and -1 * temp_angle <= ang_max:
                                angles_set.add(-1 * temp_angle)
                    else:
                        c_x = (pt_x ** 2 + pt_y ** 2 - c_y ** 2) ** 0.5
                        v2_1 = [c_x, c_y]
                        temp_angle_1 = self._get_angle(v1, v2_1)
                        v2_2 = [-1 * c_x, c_y]
                        temp_angle_2 = self._get_angle(v1, v2_2)
                        if temp_angle_1 >= ang_min and temp_angle_1 <= ang_max:
                            if temp_angle_1 not in angles_set:
                                angles_set.add(temp_angle_1)
                                if -1 * temp_angle_1 >= ang_min and -1 * temp_angle_1 <= ang_max:
                                    angles_set.add(-1 * temp_angle_1)
                        elif temp_angle_2 >= ang_min and temp_angle_2 <= ang_max:
                            if temp_angle_2 not in angles_set:
                                angles_set.add(temp_angle_2)
                                if -1 * temp_angle_2 >= ang_min and -1 * temp_angle_2 <= ang_max:
                                    angles_set.add(-1 * temp_angle_2)
                        else:
                            break
                    offset += 1

                offset = 0.5
                while True:
                    c_y = pt_y - offset
                    if pt_x ** 2 + pt_y ** 2 < c_y ** 2:
                        break
                    elif pt_x ** 2 + pt_y ** 2 == c_y ** 2:
                        c_x = 0
                        v2 = [c_x, c_y]
                        temp_angle = self._get_angle(v1, v2)
                        if temp_angle < ang_min or temp_angle > ang_max:
                            break
                        if temp_angle not in angles_set:
                            angles_set.add(temp_angle)
                            if -1 * temp_angle >= ang_min and -1 * temp_angle <= ang_max:
                                angles_set.add(-1 * temp_angle)
                    else:
                        c_x = (pt_x ** 2 + pt_y ** 2 - c_y ** 2) ** 0.5
                        v2_1 = [c_x, c_y]
                        temp_angle_1 = self._get_angle(v1, v2_1)
                        v2_2 = [-1 * c_x, c_y]
                        temp_angle_2 = self._get_angle(v1, v2_2)
                        if temp_angle_1 >= ang_min and temp_angle_1 <= ang_max:
                            if temp_angle_1 not in angles_set:
                                angles_set.add(temp_angle_1)
                                if -1 * temp_angle_1 >= ang_min and -1 * temp_angle_1 <= ang_max:
                                    angles_set.add(-1 * temp_angle_1)
                        elif temp_angle_2 >= ang_min and temp_angle_2 <= ang_max:
                            if temp_angle_2 not in angles_set:
                                angles_set.add(temp_angle_2)
                                if -1 * temp_angle_2 >= ang_min and -1 * temp_angle_2 <= ang_max:
                                    angles_set.add(-1 * temp_angle_2)
                        else:
                            break
                    offset += 1
        return list(angles_set)

    def __dotproduct(self, v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def __length(self, v):
        return math.sqrt(self.__dotproduct(v, v))

    def _get_angle(self, v1, v2):
        return math.acos(self.__dotproduct(v1, v2) /
                         (self.__length(v1) * self.__length(v2)))

    def _get_verify_angle(self, img_size, ang_range=None):
        img_height, img_width = img_size
        angles = sorted(
            self._get_crit_angles(
                img_height,
                img_width,
                ang_range=ang_range))
        angles_ub = []
        angles_lb = []
        for idx, angle in enumerate(angles):
            if angle == 0:
                mid_ang = 0.0
            elif angle < 0:
                if idx == 0:
                    if ang_range:
                        mid_ang = 0.5 * (ang_range[0] + angle)
                    else:
                        mid_ang = 0.5 * (-1 * np.pi + angle)
                else:
                    mid_ang = 0.5 * (angle + angles[idx - 1])
            else:  # angle > 0
                if idx == len(angles) - 1:
                    if ang_range:
                        mid_ang = 0.5 * (ang_range[1] + angle)
                    else:
                        mid_ang = 0.5 * (np.pi + angle)
                else:
                    mid_ang = 0.5 * (angle + angles[idx + 1])
            if mid_ang >= 0:
                angles_ub.append(mid_ang)
            else:
                angles_lb.append(mid_ang)
        angles_ub_np = np.array(angles_ub)
        angles_lb_np = np.array(angles_lb)
        angles_ub_np = np.sort(angles_ub_np)
        angles_lb_np = np.sort(angles_lb_np)[::-1]
        return angles_ub_np, angles_lb_np
