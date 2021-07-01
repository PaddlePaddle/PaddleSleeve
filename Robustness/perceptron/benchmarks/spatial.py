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

"""Metric that tests models against spatial transformations."""

import numpy as np
from tqdm import tqdm
from collections import Iterable
import math
from .base import Metric
from .base import call_decorator
import warnings


class SpatialMetric(Metric):
    """Metric that tests models against spatial transformations."""

    @call_decorator
    def __call__(self, adv, annotation=None,
                 do_rotations=True, do_translations=True,
                 x_shift_limits=(-5, 5), y_shift_limits=(-5, 5), angular_limits=None,
                 unpack=True, abort_early=True, verify=False, epsilons=1000):
        """Spatial transforms the image until it is misclassified.

        Parameters
        ----------
        adv : `numpy.ndarray`
            The original, unperturbed input as a `numpy.ndarray`.
        annotation : int
            The reference label of the original input.
        do_rotations : bool
            If False no rotations will be applied to the image
            (default True).
        do_translations : bool
            If False no translations will be applied to the image
            (default True).
        x_shift_limits : (int, int)
            Limits for horizontal translations in pixels (default (-5, 5)).
        y_shift_limits : (int, int)
            Limits for vertical translations in pixels (default (-5, 5)).
        angular_limits : (int, int)
            Limits for rotations in degrees (default None).
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        abort_early : bool
            If true, returns when got first adversarial, otherwise
            returns when all the iterations are finished.
        verify : bool
            if True, return verifiable bound
        epsilons : int or Iterable[float]
            Either Iterable of contrast levels or number of contrast
            levels between 1 and 0 that should be tried. Epsilons are
            one minus the contrast level.

        """

        if verify is True:
            warnings.warn('epsilon is not used in verification mode '
                          'and abort_early is set to True.')

        if angular_limits:
            assert len(
                angular_limits) == 2, "angular_limits has to be (range_low, range_high)"
            assert angular_limits[0] <= angular_limits[1], "angular_limits[0] should be smaller than angular_limits[1]"

        assert len(
            x_shift_limits) == 2, "x_shift_limits has to be (range_low, range_high)"
        assert x_shift_limits[0] <= x_shift_limits[1], "x_shift_limits[0] should be smaller than x_shift_limits[1]"
        assert len(
            y_shift_limits) == 2, "y_shift_limits has to be (range_low, range_high)"
        assert y_shift_limits[0] <= y_shift_limits[1], "y_shift_limits[0] should be smaller than y_shift_limits[1]"

        a = adv
        del adv
        del annotation
        del unpack

        min_, max_ = a.bounds()

        image = a.original_image
        img_height, img_width = image.shape[1:]

        if verify:
            epsilons_ub_ang, epsilons_lb_ang = self._get_verify_angle(
                image.shape[1:], ang_range=angular_limits)
            epsilons_ub_x = np.arange(
                np.max(
                    0,
                    x_shift_limits[0]),
                x_shift_limits[1],
                1)
            epsilons_lb_x = np.arange(
                np.min(0, x_shift_limits[1]), x_shift_limits[0], -1)
            epsilons_ub_y = np.arange(
                np.max(
                    0,
                    y_shift_limits[0]),
                y_shift_limits[1],
                1)
            epsilons_lb_y = np.arange(
                np.min(0, y_shift_limits[1]), y_shift_limits[0], -1)

        elif not isinstance(epsilons, Iterable):
            if not angular_limits:
                ang_max = np.pi
                ang_min = -1 * np.pi
            else:
                ang_max = angular_limits[1]
                ang_min = angular_limits[0]

            if ang_min >= 0:
                epsilons_ub_ang = np.linspace(
                    ang_min, ang_max, num=epsilons + 1)[1:]
                epsilons_lb_ang = []
            elif ang_max <= 0:
                epsilons_ub_ang = []
                epsilons_lb_ang = np.linspace(
                    ang_max, ang_min, num=epsilons + 1)[1:]
            else:
                epsilons_ub_ang = np.linspace(
                    0, ang_max, num=int(epsilons / 2 + 1))[1:]
                epsilons_lb_ang = np.linspace(
                    0, ang_min, num=int(epsilons / 2 + 1))[1:]

            if x_shift_limits[0] >= 0:
                epsilons_ub_x = np.linspace(
                    x_shift_limits[0], x_shift_limits[1], num=epsilons + 1)[1:]
                epsilons_lb_x = []
            elif x_shift_limits[1] <= 0:
                epsilons_ub_x = []
                epsilons_lb_x = np.linspace(
                    x_shift_limits[1], x_shift_limits[0], num=epsilons + 1)[1:]
            else:
                epsilons_ub_x = np.linspace(
                    0, x_shift_limits[1], num=int(epsilons / 2 + 1))[1:]
                epsilons_lb_x = np.linspace(
                    0, x_shift_limits[0], num=int(epsilons / 2 + 1))[1:]

            if y_shift_limits[0] >= 0:
                epsilons_ub_y = np.linspace(
                    y_shift_limits[0], y_shift_limits[1], num=epsilons + 1)[1:]
                epsilons_lb_y = []
            elif y_shift_limits[1] <= 0:
                epsilons_ub_y = []
                epsilons_lb_y = np.linspace(
                    y_shift_limits[1], y_shift_limits[0], num=epsilons + 1)[1:]
            else:
                epsilons_ub_y = np.linspace(
                    0, y_shift_limits[1], num=int(epsilons / 2 + 1))[1:]
                epsilons_lb_y = np.linspace(
                    0, y_shift_limits[0], num=int(epsilons / 2 + 1))[1:]
        else:
            epsilons_ub_ang = epsilons
            epsilons_lb_ang = []
            epsilons_ub_x = epsilons
            epsilons_lb_x = []
            epsilons_ub_y = epsilons
            epsilons_lb_y = []

        epsilons_ub_x = np.unique(epsilons_ub_x.astype(int))
        epsilons_lb_x = np.flip(np.unique(epsilons_lb_x.astype(int)), -1)
        epsilons_ub_y = np.unique(epsilons_ub_y.astype(int))
        epsilons_lb_y = np.flip(np.unique(epsilons_lb_y.astype(int)), -1)

        if not do_rotations:
            epsilons_ub_ang = [0.]
            epsilons_lb_ang = []
        if not do_translations:
            epsilons_ub_x = [0]
            epsilons_lb_x = []
            epsilons_ub_y = [0]
            epsilons_lb_y = []

        upper_bounds = [0]
        lower_bounds = [0]
        image_cv = np.transpose(image, (1, 2, 0))
        print('Generating adversarial examples.')
        is_break = False
        for _, epsilon_x in enumerate(tqdm(epsilons_ub_x)):
            if is_break:
                break
            for _, epsilon_y in enumerate(epsilons_ub_y):
                if is_break:
                    break
                for _, epsilon_ang in enumerate(epsilons_ub_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        upper_bounds.append(epsilon_ang)

                for _, epsilon_ang in enumerate(epsilons_lb_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        lower_bounds.append(epsilon_ang)
            for _, epsilon_y in enumerate(epsilons_lb_y):
                if is_break:
                    break
                for _, epsilon_ang in enumerate(epsilons_ub_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        upper_bounds.append(epsilon_ang)

                for _, epsilon_ang in enumerate(epsilons_lb_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        lower_bounds.append(epsilon_ang)

        for _, epsilon_x in enumerate(tqdm(epsilons_lb_x)):
            if is_break:
                break
            for _, epsilon_y in enumerate(epsilons_ub_y):
                if is_break:
                    break
                for _, epsilon_ang in enumerate(epsilons_ub_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        upper_bounds.append(epsilon_ang)

                for _, epsilon_ang in enumerate(epsilons_lb_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        lower_bounds.append(epsilon_ang)
            for _, epsilon_y in enumerate(epsilons_lb_y):
                if is_break:
                    break
                for _, epsilon_ang in enumerate(epsilons_ub_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        upper_bounds.append(epsilon_ang)

                for _, epsilon_ang in enumerate(epsilons_lb_ang):
                    perturbed = self._spatial(
                        image_cv, epsilon_x, epsilon_y, epsilon_ang)
                    perturbed = np.transpose(perturbed, (2, 0, 1))
                    _, is_adversarial = a.predictions(perturbed)
                    if is_adversarial:
                        if abort_early or verify:
                            if not verify:
                                is_break = True
                            break
                    else:
                        lower_bounds.append(epsilon_ang)

        a.verifiable_bounds = (max(upper_bounds), min(lower_bounds))

        return

    def _spatial(self, img_cv, x_shift, y_shift, rotate_ang):
        import cv2
        img_height, img_width = img_cv.shape[:2]
        M = cv2.getRotationMatrix2D(
            (img_width / 2., img_height / 2.), rotate_ang * 180 / np.pi, 1)
        rotate = cv2.warpAffine(img_cv, M, (img_width, img_height))
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        rotate_translate = cv2.warpAffine(rotate, M, (img_width, img_height))
        return rotate_translate

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
