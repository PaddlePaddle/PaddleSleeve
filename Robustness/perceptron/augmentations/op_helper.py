# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
# this file contains helper methods for common image operations

import cv2
import numpy as np
from scipy.ndimage import zoom as scizoom


def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        coords = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    aliased_disk = np.asarray((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3, rng=None):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100
    if rng is None:
        rng = np.random.default_rng()

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def motion_Kernel(dim, angle):
    if isinstance(dim, int):
        dim = (dim, dim)
    dim = np.array(dim).astype(int)
    center = (dim - 1) / 2
    kernel = np.zeros(dim)
    angle = np.fmod(angle, 180.0)
    if angle == 90:
        kernel[:, int(center[1])] = 1
        return kernel

    delta_x = 0
    delta_y = 0
    while True:
        if (center[1] + delta_x).is_integer():
            if int(round(center[0] + delta_y)) < 0 or int(round(center[0] + delta_y)) >= dim[0] or int(
                    center[1] + delta_x) < 0 or int(center[1] + delta_x) >= dim[1]:
                break

            kernel[int(round(center[0] - delta_y)),
                   int(center[1] + delta_x)] = 1
            kernel[int(round(center[0] + delta_y)),
                   int(center[1] - delta_x)] = 1
        delta_x += 0.5
        delta_y = delta_x * np.tan(angle / 180 * np.pi)
    normalizationFactor = np.count_nonzero(kernel)
    kernel /= normalizationFactor
    return kernel
