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

"""Metric that tests models against fog variations."""

import numpy as np
from tqdm import tqdm
from collections import Iterable
from .base import Metric
from .base import call_decorator
import warnings


class FogMetric(Metric):
    """Metric that tests models against fog variations."""

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, verify=False, epsilons=1000):
        """Change the fog of the image until it is misclassified.

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
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        img_height, img_width = hw

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons)[1:]
        else:
            epsilons = epsilons

        cc0 = [1.0, 3.5]
        cc1 = [2.3, 1.0]
        map_size = 2
        while map_size < max(img_height, img_width):
            map_size *= 2
        max_val = image.max()

        for _, epsilon in enumerate(tqdm(epsilons)):
            p0 = cc0[0] + epsilon * (cc0[1] - cc0[0])
            p1 = cc1[0] + epsilon * (cc1[1] - cc1[0])
            perturbed = image + p0 * np.expand_dims(
                plasma_fractal(
                    mapsize=map_size, wibbledecay=p1)
                [:img_height, :img_width], 0)
            perturbed = np.clip(perturbed * max_val /
                                (max_val + p0), min_, max_) * max_
            perturbed = perturbed.astype(np.float32)

            _, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                if abort_early or verify:
                    break
            else:
                bound = epsilon
                a.verifiable_bounds = (bound, None)

        return


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in
    range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        """wibbled mean"""
        return array / 4 + wibble * np.random.uniform(-wibble,
                                                      wibble,
                                                      array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref,
                                          shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize
                          // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize //
                 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
                 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()
