# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# operators to augment an input image
# e.g. blur, zoom, rotate, compress, etc.
from __future__ import division

import uuid

import numpy
import numpy as np
import cv2
import skimage
from skimage import color
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from perceptron.augmentations.op_helper import disk, plasma_fractal, motion_Kernel

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


registered_ops = []
np.random.seed(0)


def register_op(cls):
    registered_ops.append(cls.__name__)
    if not hasattr(BaseOperator, cls.__name__):
        setattr(BaseOperator, cls.__name__, cls)
    else:
        raise KeyError("The {} class has been registered.".format(cls.__name__))
    return cls


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, **kwargs):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, format='CHW', **kwargs):
        """ Process a sample.
        Args:
            sample (np.ndarray): the sample image
            context (dict): info about this sample processing
        Returns:
            result (np.ndarray): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)
    

class ImageOperator(BaseOperator):
    """
    Base operator that takes an image as its input
    """
    def __init__(self, name=None, format='CHW', bound=(0, 1)):
        super(ImageOperator, self).__init__(name=name)
        self._format = format
        self._bound = bound

    def apply(self, sample, **kwargs):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, **kwargs):
        """ Process a sample.
        Args:
            sample (np.ndarray): the sample image
            context (dict): info about this sample processing
        Returns:
            result (np.ndarray): a processed sample
        """
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                img = sample[i]
                if not isinstance(img, np.ndarray):
                    img = img.numpy()
                if self._format == 'CHW':
                    img = np.transpose(img, [1, 2, 0])
                if self._bound == (0, 1):
                    img = (img * 255).astype('uint8')

                img = self.apply(img.astype('uint8'), **kwargs)

                if self._format == 'CHW':
                    img = np.transpose(img, [2, 0, 1])
                if self._bound == (0, 1):
                    img = img / 255
                sample[i] = img
        else:
            img = sample
            if not isinstance(img, np.ndarray):
                img = img.numpy()
            if self._format == 'CHW':
                img = np.transpose(img, [1, 2, 0])
            if self._bound == (0, 1):
                img = (img * 255).astype('uint8')

            img = self.apply(img.astype('uint8'), **kwargs)

            if self._format == 'CHW':
                img = np.transpose(img, [2, 0, 1])
            if self._bound == (0, 1):
                img = img / 255
            sample = img
        return sample


class ArrayOperator(BaseOperator):
    """
    Base operator that operate on an array
    """
    def __init__(self, name=None, format='CHW', bound=(0, 1)):
        super(ArrayOperator, self).__init__(name=name)
        self._format = format
        self._bound = bound

    def apply(self, sample, **kwargs):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, **kwargs):
        """ Process a sample.
        Args:
            sample (np.ndarray): the sample image
            context (dict): info about this sample processing
        Returns:
            result (np.ndarray): a processed sample
        """
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                img = sample[i]
                if not isinstance(img, np.ndarray):
                    img = img.numpy()
                if self._format != 'CHW':
                    img = np.transpose(img, [2, 0, 1])
                if self._bound != (0, 1):
                    img = img / 255

                img = self.apply(img, **kwargs)

                if self._format != 'CHW':
                    img = np.transpose(img, [1, 2, 0])
                if self._bound != (0, 1):
                    img = (img * 255).astype('uint8')
                sample[i] = img
        else:
            img = sample
            if not isinstance(img, np.ndarray):
                img = img.numpy()
            if self._format != 'CHW':
                img = np.transpose(img, [2, 0, 1])
            if self._bound != (0, 1):
                img = img / 255

            img = self.apply(img, **kwargs)

            if self._format != 'CHW':
                img = np.transpose(img, [1, 2, 0])
            if self._bound != (0, 1):
                img = (img * 255).astype('uint8')
            sample = img
        return sample


@register_op
class Curve(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Curve, self).__init__(format=format, bound=bound)
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def apply(self, img, mag=0):
        orig_h, orig_w, c = img.shape
        side = max(orig_h, orig_w)

        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_CUBIC)

        w = side
        h = side
        w_25 = 0.25 * w
        w_50 = 0.50 * w
        w_75 = 0.75 * w

        b = [1.1, .95, .8]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        rmin = b[index]

        r = np.random.uniform(rmin, rmin + .1) * h
        x1 = (r ** 2 - w_50 ** 2) ** 0.5
        h1 = r - x1

        t = np.random.uniform(0.4, 0.5) * h

        w2 = w_50 * t / r
        hi = x1 * t / r
        h2 = h1 + hi

        sinb_2 = ((1 - x1 / r) / 2) ** 0.5
        cosb_2 = ((1 + x1 / r) / 2) ** 0.5
        w3 = w_50 - r * sinb_2
        h3 = r - r * cosb_2

        w4 = w_50 - (r - t) * sinb_2
        h4 = r - (r - t) * cosb_2

        w5 = 0.5 * w2
        h5 = h1 + 0.5 * hi
        h_50 = 0.50 * h

        srcpt = [(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h),
                 (0, h_50), (w, h_50)]
        dstpt = [(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4),
                 (w - w4, h4), (w5, h5), (w - w5, h5)]

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)

        img = img[:side//2, :, :]
        img = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return img


@register_op
class Distort(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Distort, self).__init__(format=format, bound=bound)
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def apply(self, img, mag=0):
        h, w, c = img.shape

        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # top pts
        srcpt.append([p, p])
        x = np.random.uniform(0, frac) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt.append([p + x, p + y])

        srcpt.append([p + w_33, p])
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt.append([p + w_33 + x, p + y])

        srcpt.append([p + w_66, p])
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt.append([p + w_66 + x, p + y])

        srcpt.append([w - p, p])
        x = np.random.uniform(-frac, 0) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt.append([w - p + x, p + y])

        # bottom pts
        srcpt.append([p, h - p])
        x = np.random.uniform(0, frac) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt.append([p + x, h - p + y])

        srcpt.append([p + w_33, h - p])
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt.append([p + w_33 + x, h - p + y])

        srcpt.append([p + w_66, h - p])
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt.append([p + w_66 + x, h - p + y])

        srcpt.append([w - p, h - p])
        x = np.random.uniform(-frac, 0) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt.append([w - p + x, h - p + y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)

        return img


@register_op
class Stretch(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Stretch, self).__init__(format=format, bound=bound)
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def apply(self, img, mag=0):
        h, w, c = img.shape

        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([p, p])
        srcpt.append([p, h - p])
        srcpt.append([p, h_50])
        x = np.random.uniform(0, frac) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt.append([p + x, p])
        dstpt.append([p + x, h - p])
        dstpt.append([p + x, h_50])

        # 2nd left-most
        srcpt.append([p + w_33, p])
        srcpt.append([p + w_33, h - p])
        x = np.random.uniform(-frac, frac) * w_33
        dstpt.append([p + w_33 + x, p])
        dstpt.append([p + w_33 + x, h - p])

        # 3rd left-most
        srcpt.append([p + w_66, p])
        srcpt.append([p + w_66, h - p])
        x = np.random.uniform(-frac, frac) * w_33
        dstpt.append([p + w_66 + x, p])
        dstpt.append([p + w_66 + x, h - p])

        # right-most
        srcpt.append([w - p, p])
        srcpt.append([w - p, h - p])
        srcpt.append([w - p, h_50])
        x = np.random.uniform(-frac, 0) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt.append([w - p + x, p])
        dstpt.append([w - p + x, h - p])
        dstpt.append([w - p + x, h_50])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)

        return img


@register_op
class Shrink(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Shrink, self).__init__(format=format, bound=bound)
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def apply(self, img, mag=0):
        h, w, c = img.shape
        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([p, p])
        srcpt.append([p, h - p])
        x = np.random.uniform(frac - .1, frac) * w_33
        y = np.random.uniform(frac - .1, frac) * h_50
        dstpt.append([p + x, p + y])
        dstpt.append([p + x, h - p - y])

        # 2nd left-most
        srcpt.append([p + w_33, p])
        srcpt.append([p + w_33, h - p])
        dstpt.append([p + w_33, p + y])
        dstpt.append([p + w_33, h - p - y])

        # 3rd left-most
        srcpt.append([p + w_66, p])
        srcpt.append([p + w_66, h - p])
        dstpt.append([p + w_66, p + y])
        dstpt.append([p + w_66, h - p - y])

        # right-most
        srcpt.append([w - p, p])
        srcpt.append([w - p, h - p])
        dstpt.append([w - p - x, p + y])
        dstpt.append([w - p - x, h - p - y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)

        return img


@register_op
class Rotate(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Rotate, self).__init__(format=format, bound=bound)
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def apply(self, img, mag=0):
        h, w, c = img.shape
        side = max(h, w)
        img = Image.fromarray(img)
        img = img.resize((side, side), Image.BICUBIC)

        b = [10, 20, 30]
        if mag < 0 or mag >= len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = np.random.uniform(rotate_angle - 15, rotate_angle)
        if np.random.uniform(0, 1) < 0.5:
            angle = -angle

        img = img.rotate(angle=angle, resample=Image.BICUBIC)
        img = img.resize((w, h), Image.BICUBIC)
        img = np.asarray(img)

        return img


@register_op
class Perspective(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Perspective, self).__init__(format=format, bound=bound)
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def apply(self, img, mag=0):
        h, w, c = img.shape

        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        b = [.05, .1, .15]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        low = b[index]

        high = 1 - low
        if np.random.uniform(0, 1) > 0.5:
            topright_y = np.random.uniform(low, low + .1) * h
            bottomright_y = np.random.uniform(high - .1, high) * h
            dest = np.float32([[0, 0], [w, topright_y], [0, h], [w, bottomright_y]])
        else:
            topleft_y = np.random.uniform(low, low + .1) * h
            bottomleft_y = np.random.uniform(high - .1, high) * h
            dest = np.float32([[0, topleft_y], [w, 0], [0, bottomleft_y], [w, h]])
        M = cv2.getPerspectiveTransform(src, dest)
        img = cv2.warpPerspective(img, M, (w, h))

        return img
    

@register_op
class VFlip(ArrayOperator):
    def __init__(self, format='CHW', bound=(0,1)):
        super(VFlip, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c, h, w = img.shape
        pert = np.zeros_like(img)
        for i in range(h):
            pert[:, i, :] = img[:, h-i-1, :]
        return pert


@register_op
class HFlip(ArrayOperator):
    def __init__(self, format='CHW', bound=(0,1)):
        super(HFlip, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c, h, w = img.shape
        pert = np.zeros_like(img)
        for i in range(w):
            pert[:, :, i] = img[:, :, w-1-i]
        return pert


@register_op
class VGrid(ImageOperator):
    def __init__(self, format='CHW', bound=(0,1), max_width=4, copy=False):
        super(VGrid, self).__init__(format=format, bound=bound)
        self._copy = copy
        self._max = max_width

    def apply(self, img, mag=0):
        img = Image.fromarray(img)

        if self._copy:
            img = img.copy()
        w, h = img.size

        if mag < 0 or mag > self._max:
            line_width = np.random.randint(1, self._max)
            image_stripe = np.random.randint(1, self._max)
        else:
            line_width = 1
            image_stripe = 4 - mag

        n_lines = w // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            x = image_stripe * i + line_width * (i - 1)
            draw.line([(x, 0), (x, h)], width=line_width, fill='black')

        return np.asarray(img)


@register_op
class HGrid(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1), max_width=4, copy=False):
        super(HGrid, self).__init__(format=format, bound=bound)
        self._copy = copy
        self._max = max_width

    def apply(self, img, mag=0):
        img = Image.fromarray(img)

        if self._copy:
            img = img.copy()
        w, h = img.size
        if mag < 0 or mag > self._max:
            line_width = np.random.randint(1, self._max)
            image_stripe = np.random.randint(1, self._max)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = h // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            y = image_stripe * i + line_width * (i - 1)
            draw.line([(0, y), (w, y)], width=line_width, fill='black')

        return np.asarray(img)


@register_op
class RectGrid(ImageOperator):
    def __init__(self, format='CHW', bound=(0,1), ellipse=False):
        super(RectGrid, self).__init__(format=format, bound=bound)
        self.isellipse = ellipse

    def apply(self, img, mag=0):
        img = Image.fromarray(img)
        w, h = img.size

        line_width = 1
        image_stripe = 3 - mag  # self.rng.integers(2, 6)
        offset = 4 if self.isellipse else 1
        n_lines = ((h // 2) // (line_width + image_stripe)) + offset
        draw = ImageDraw.Draw(img)
        x_center = w // 2
        y_center = h // 2
        for i in range(1, n_lines):
            dx = image_stripe * i + line_width * (i - 1)
            dy = image_stripe * i + line_width * (i - 1)
            x1 = x_center - (dx * w // h)
            y1 = y_center - dy
            x2 = x_center + (dx * w / h)
            y2 = y_center + dy
            if self.isellipse:
                draw.ellipse([(x1, y1), (x2, y2)], width=line_width, outline='black')
            else:
                draw.rectangle([(x1, y1), (x2, y2)], width=line_width, outline='black')

        return np.asarray(img)


@register_op
class GaussianBlur(ImageOperator):
    def __init__(self, format='CHW', bound=(0,1)):
        super(GaussianBlur, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        h, w, c = img.shape
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)
        sigmas = [.5, 1, 2]
        if mag < 0 or mag >= len(sigmas):
            index = self.rng.integers(0, len(sigmas))
        else:
            index = mag

        sigma = sigmas[index]
        img = cv2.GaussianBlur(img, kernel, sigma, sigma)
        return img


@register_op
class MedianBlur(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(MedianBlur, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        h, w, c = img.shape
        ksize = [3, 5, 7]
        if mag < 0 or mag >= len(ksize):
            index = self.rng.integers(0, len(ksize))
        else:
            index = mag

        ksize = ksize[index]
        img = cv2.medianBlur(img, ksize)
        return img


@register_op
class DefocusBlur(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(DefocusBlur, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        n_channels, h, w = img.shape
        isgray = n_channels == 1
        c = [(2, 0.1), (8, 0.15), (10, 0.2)]  # , (6, 0.5)] #prev 2 levels only
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        if isgray:
            img = np.repeat(img, 3, axis=0)
            n_channels = 3
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(n_channels):
            channels.append(cv2.filter2D(img[d, :, :], -1, kernel))
        channels = np.stack(channels)

        img = np.clip(channels, 0, 1)
        if isgray:
            img = img[:1, :, :]

        return img


@register_op
class MotionBlur(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(MotionBlur, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        eps = [5, 10, 20]
        if mag < 0 or mag >= len(eps):
            index = np.random.randint(0, len(eps))
        else:
            index = mag
        degree = eps[index]
        angle = np.random.uniform(-45, 45)
        kernel = motion_Kernel((degree, degree), angle)
        img = np.transpose(img, [1, 2, 0])

        blurred = cv2.filter2D(img, -1, kernel)
        blurred = np.transpose(blurred, [2, 0, 1])

        return np.clip(blurred, 0, 1)


@register_op
class GlassBlur(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(GlassBlur, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        from skimage import filters as filter
        n_channels, h, w = img.shape
        c = [(0.45, 1, 1), (0.6, 1, 2), (0.75, 1, 2)]  # , (1, 2, 3)] #prev 2 levels only
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag

        c = c[index]

        for i in range(n_channels):
            img[i] = filter.gaussian(img[i], sigma=c[0], multichannel=False)

        # locally shuffle pixels
        for i in range(c[2]):
            for y in range(h - c[1], c[1], -1):
                for x in range(w - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    y_prime, x_prime = y + dy, x + dx
                    # swap
                    img[:, y, x], img[:, y_prime, x_prime] = img[:, y_prime, x_prime], img[:, y, x]

        for i in range(n_channels):
            img[i] = np.clip(filter.gaussian(img[i], sigma=c[0], multichannel=False), 0, 1)

        return img


@register_op
class ZoomBlur(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(ZoomBlur, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        h, w, n_channels = img.shape
        c = [np.arange(1, 1.11, .01),
             np.arange(1, 1.16, .01),
             np.arange(1, 1.21, .02)]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag

        c = c[index]

        uint8_img = img
        img = img / 255

        out = np.zeros_like(img)
        for zoom_factor in c:
            zw = int(w * zoom_factor)
            zh = int(h * zoom_factor)
            zoom_img = cv2.resize(uint8_img, (zw, zh), interpolation=cv2.INTER_CUBIC)
            x1 = (zw - w) // 2
            y1 = (zh - h) // 2
            x2 = x1 + w
            y2 = y1 + h
            zoom_img = zoom_img[y1:y2, x1:x2]
            out += (np.asarray(zoom_img) / 255.).astype(np.float32)

        img = (img + out) / (len(c) + 1)
        img = (np.clip(img, 0, 1) * 255).astype('uint8')

        return img


@register_op
class JPEG_Compression(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(JPEG_Compression, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        rate = [90, 70, 50]
        if mag < 0 or mag >= len(rate):
            index = 0
        else:
            index = mag

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rate = rate[index]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rate]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

        return img


@register_op
class GaussianNoise(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(GaussianNoise, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        b = [.06, 0.09, 0.12]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + 0.03)
        img = np.clip(img + np.random.normal(size=img.shape, scale=c), 0, 1)
        return img


@register_op
class ShotNoise(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(ShotNoise, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        b = [13, 8, 3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + 7)
        img = np.clip(np.random.poisson(img * c) / float(c), 0, 1)
        return img


@register_op
class ImpulseNoise(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(ImpulseNoise, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        b = [.03, .07, .11]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + .04)

        img = skimage.util.random_noise(np.asarray(img) / 255., mode='s&p', amount=c) * 255
        return img


@register_op
class SpeckleNoise(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(SpeckleNoise, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        b = [.15, .2, .25]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + .05)
        img = np.clip(img + img * np.random.normal(size=img.shape, scale=c), 0, 1)
        return img


@register_op
class Fog(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Fog, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        n_channels, h, w = img.shape
        c = [(1.5, 2), (2., 2), (2.5, 1.7)]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        isgray = n_channels == 1

        max_val = img.max()
        # Make sure fog image is at least twice the size of the input image
        max_size = int(2 ** np.ceil(np.log2(max(w, h)) + 1))
        fog = c[0] * plasma_fractal(mapsize=max_size, wibbledecay=c[1])[:h, :w][np.newaxis, ...]
        if not isgray:
            fog = np.repeat(fog, 3, axis=0)

        img += fog
        img = np.clip(img * max_val / (max_val + c[0]), 0, 1)
        return img


@register_op
class Rain(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Rain, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        h, w, n_channels = img.shape
        isgray = n_channels == 1
        line_width = np.random.randint(1, 2)

        c = [50, 100, 150]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        n_rains = np.random.randint(c, c + 20)
        slant = np.random.randint(-60, 60)
        fillcolor = 200 if isgray else (200, 200, 200)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        max_length = min(w, h, 10)
        for i in range(1, n_rains):
            length = np.random.randint(5, max_length)
            x1 = np.random.randint(0, w - length)
            y1 = np.random.randint(0, h - length)
            x2 = x1 + length * np.sin(slant * np.pi / 180.)
            y2 = y1 + length * np.cos(slant * np.pi / 180.)
            x2 = int(x2)
            y2 = int(y2)
            draw.line([(x1, y1), (x2, y2)], width=line_width, fill=fillcolor)

        return np.asarray(img)


@register_op
class Shadow(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Shadow, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        h, w, n_channels = img.shape
        isgray = n_channels == 1

        c = [64, 96, 128]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        img = Image.fromarray(img)
        img = img.convert('RGBA')
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        transparency = np.random.randint(c, c + 32)
        x1 = np.random.randint(0, w // 2)
        y1 = 0
        x2 = np.random.randint(w // 2, w)
        y2 = 0
        x3 = np.random.randint(w // 2, w)
        y3 = h - 1
        x4 = np.random.randint(0, w // 2)
        y4 = h - 1

        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=(0, 0, 0, transparency))

        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        if isgray:
            img = ImageOps.grayscale(img)

        return np.asarray(img)

@register_op
class Snow(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Snow, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        min_, max_ = (0, 1)
        c, img_height, img_width = img.shape

        epsilons = [1, 2, 4]
        if mag < 0 or mag >= len(epsilons):
            index = 0
        else:
            index = mag
        epsilon = epsilons[index]
        angle = np.random.randint(-45, 45)

        snow_mask_np = np.zeros((img_height // 10, img_height // 10, 3))
        ch = snow_mask_np.shape[0] // 2
        cw = snow_mask_np.shape[1] // 2
        cr = min(img_height, img_width) * 0.1
        for i in range(snow_mask_np.shape[0]):
            for j in range(snow_mask_np.shape[1]):
                if (i - ch) ** 2 + (j - cw) ** 2 <= cr:
                    snow_mask_np[i, j] = np.ones(3)

        kernel = motion_Kernel((int(ch * 0.9), int(cw * 0.9)), angle)
        blured = cv2.filter2D(snow_mask_np, -1, kernel)
        blured = np.clip(blured, min_, max_).astype(np.float32)
        blured = blured * max_
        blured_h, blured_w = blured.shape[:2]
        blured = np.transpose(blured, (2, 0, 1))

        p0 = int(1 + epsilon * 50)
        positions_h = np.random.randint(img_height - blured_h, size=p0)
        positions_w = np.random.randint(img_width - blured_w, size=p0)
        for temp_h, temp_w in zip(positions_h, positions_w):
            img[:, temp_h: temp_h + blured_h, temp_w: temp_w + blured_w] += blured
        img = np.clip(img, min_, max_)

        return img


@register_op
class Contrast(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Contrast, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        means = np.mean(img, axis=(1, 2), keepdims=True)
        img = np.clip((img - means) * c + means, 0, 1)

        return img


@register_op
class Brightness(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Brightness, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c = [.1, .2, .3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels, h, w = img.shape
        isgray = n_channels == 1

        if isgray:
            img = np.repeat(img, 3, axis=0)

        img = np.transpose(img, [1, 2, 0])
        img = color.rgb2hsv(img)
        img[:, :, 2] = np.clip(img[:, :, 2] + c, 0, 1)
        img = color.hsv2rgb(img)
        img = np.transpose(img, [2, 0, 1])

        return np.clip(img, 0, 1)


@register_op
class Pixelate(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Pixelate, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c, h, w = img.shape
        p_size = [2, 3, 5]
        if mag < 0 or mag >= len(p_size):
            index = np.random.randint(0, len(p_size))
        else:
            index = mag
        p_size = p_size[index]
        blocks = h // p_size
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
        # 在x和y方向上循环遍历块
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                roi = img[:, startY:endY, startX:endX]
                img[:, startY:endY, startX:endX] = np.mean(roi, axis=(1, 2), keepdims=True)

        return img


@register_op
class MaxSmoothing(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(MaxSmoothing, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c, h, w = img.shape
        p_size = [2, 3, 5]
        if mag < 0 or mag >= len(p_size):
            index = np.random.randint(0, len(p_size))
        else:
            index = mag
        p_size = p_size[index]
        blocks = h // p_size
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")

        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                roi = img[:, startY:endY, startX:endX]
                img[:, startY:endY, startX:endX] = np.max(roi, axis=(1, 2), keepdims=True)

        return img


@register_op
class AvgSmoothing(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(AvgSmoothing, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        import paddle
        c, h, w = img.shape
        k = [(3, 1), (5, 2), (7, 3)]
        if mag < 0 or mag >= len(k):
            index = np.random.randint(0, len(k))
        else:
            index = mag
        k = k[index]
        img = paddle.unsqueeze(paddle.to_tensor(img), axis=0)
        img = paddle.nn.AvgPool2D(stride=1, kernel_size=k[0], padding=k[1])(img)
        return paddle.squeeze(img).numpy()


@register_op
class Posterize(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Posterize, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c = [6, 3, 1]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        bit = np.random.randint(c, c + 2)
        img = Image.fromarray(img)
        img = ImageOps.posterize(img, bit)

        return np.asarray(img)


@register_op
class Solarize(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Solarize, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c = [192, 128, 64]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        thresh = np.random.randint(c, c + 64)
        img = Image.fromarray(img)
        img = ImageOps.solarize(img, thresh)

        return np.asarray(img)


@register_op
class Invert(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Invert, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        return 1 - img


@register_op
class Equalize(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Equalize, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        img = Image.fromarray(img)
        img = ImageOps.equalize(img)

        return np.asarray(img)


@register_op
class Sharpness(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Sharpness, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c = [.1, .7, 1.3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c + .6)
        img = Image.fromarray(img)
        img = ImageEnhance.Sharpness(img).enhance(magnitude)

        return np.asarray(img)


@register_op
class Color(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Color, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        c = [.1, .8, 1.5]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c + .6)
        img = Image.fromarray(img)
        img = ImageEnhance.Color(img).enhance(magnitude)

        return np.asarray(img)


@register_op
class HueSaturation(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(HueSaturation, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        h, w, n_channels = img.shape

        c = [(5, 20, 5), (10, 30, 10), (20, 40, 20)]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        hue_shift = np.random.randint(-c[0], c[0])
        sat_shift = np.random.randint(-c[1], c[1])
        val_shift = np.random.randint(-c[2], c[2])

        if n_channels == 1:
            hue_shift = 0
            sat_shift = 0
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if hue_shift != 0:
            lut_hue = np.arange(0, 256, dtype=np.int16)
            lut_hue = np.mod(lut_hue + hue_shift, 180).astype('uint8')
            hue = cv2.LUT(hue, lut_hue)

        if sat_shift != 0:
            lut_sat = np.arange(0, 256, dtype=np.int16)
            lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype('uint8')
            sat = cv2.LUT(sat, lut_sat)

        if val_shift != 0:
            lut_val = np.arange(0, 256, dtype=np.int16)
            lut_val = np.clip(lut_val + val_shift, 0, 255).astype('uint8')
            val = cv2.LUT(val, lut_val)

        img = cv2.merge((hue, sat, val)).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        if n_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img


@register_op
class Transpose(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(Transpose, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        return np.transpose(img, [0, 2, 1])


@register_op
class BitReduction(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(BitReduction, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        bit_len = [6, 4, 2]
        if mag < 0 or mag >= len(bit_len):
            index = np.random.randint(0, len(bit_len))
        else:
            index = mag
        bit_len = bit_len[index]
        max_num = 2 ** bit_len - 1
        img = np.rint(img * max_num)
        img = img / max_num

        return img

@register_op
class GridDistortion(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(GridDistortion, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        height, width = img.shape[:2]
        num_steps = 10
        eps = [0.3, 0.5, 0.7]
        eps = eps[int(np.clip(mag, 0, 2))]
        xsteps = np.random.uniform(1-eps, 1+eps, size=num_steps)
        ysteps = np.random.uniform(1-eps, 1+eps, size=num_steps)

        x_step = width // num_steps
        xx = np.zeros(width, np.float32)
        prev = 0
        for idx in range(num_steps + 1):
            x = idx * x_step
            start = int(x)
            end = int(x) + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step * xsteps[idx]

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        y_step = height // num_steps
        yy = np.zeros(height, np.float32)
        prev = 0
        for idx in range(num_steps + 1):
            y = idx * y_step
            start = int(y)
            end = int(y) + y_step
            if end > height:
                end = height
                cur = height
            else:
                cur = prev + y_step * ysteps[idx]

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        img = cv2.remap(img, map1=map_x, map2=map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                        borderValue=None)

        return img


@register_op
class OpticalDistortion(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(OpticalDistortion, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        height, width = img.shape[:2]
        eps = [0.3, 0.5, 0.7]
        eps = eps[int(np.clip(mag, 0, 2))]
        k, dx, dy = np.random.uniform(1 - eps, 1 + eps, size=3)

        fx = width
        fy = height
        cx = width * 0.5 + dx
        cy = height * 0.5 + dy

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
        img = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
            borderValue=None,
        )
        return img


@register_op
class Translation(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1), border=None):
        super(Translation, self).__init__(format=format, bound=bound)
        self._border = border

    def apply(self, img, mag=0):
        h, w, c = img.shape
        eps = [0.1, 0.3, 0.5]
        eps = eps[int(np.clip(mag, 0, 2))]
        dx, dy = np.random.uniform(-eps, eps, size=2)

        tx = w * dx
        ty = h * dy
        T = np.array([[1, 0, tx], [0, 1, ty]], dtype='float32')
        if self._border == 'wrap':
            border = cv2.BORDER_WRAP
        elif self._border == 'reflect':
            border = cv2.BORDER_REFLECT_101
        else:
            border = cv2.BORDER_CONSTANT
        img = cv2.warpAffine(img, T, (w, h), borderMode=border)
        return img


@register_op
class RandomCrop(ImageOperator):
    def __init__(self, format='CHW', bound=(0, 1)):
        super(RandomCrop, self).__init__(format=format, bound=bound)

    def apply(self, img, mag=0):
        h, w, c = img.shape
        eps = [0.1, 0.2, 0.3]
        eps = eps[int(np.clip(mag, 0, 2))]
        dx1, dx2, dy1, dy2 = np.clip(np.random.uniform(-eps, eps, size=4), 0, eps)

        x1 = int(w * dx1)
        x2 = int(w * (1 - dx2))
        y1 = int(h * dy1)
        y2 = int(h * (1 - dy2))
        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (w, h))
        return img


@register_op
class RandomMask(ArrayOperator):
    def __init__(self, format='CHW', bound=(0, 1), pattern=None):
        super(RandomMask, self).__init__(format=format, bound=bound)
        self._pattern = pattern

    def apply(self, img, mag=0):
        c, h, w = img.shape
        eps = [0.1, 0.2, 0.3]
        eps = eps[int(np.clip(mag, 0, 2))]
        dw, dh = np.random.uniform(0, eps, size=2)
        dw = int(w * dw)
        dh = int(h * dh)
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        x2 = min(x1 + dw, w)
        y2 = min(y1 + dh, h)
        dh = int(y2 - y1)
        dw = int(x2 - x1)
        if self._pattern == 'gaussian':
            mask = np.random.normal(0.5, size=(3, dh, dw))
        elif self._pattern == 'random':
            mask = np.random.uniform() * np.ones((3, dh, dw))
        else:
            mask = np.ones((3, dh, dw)) * 0.7
        img[:, y1:y2, x1:x2] = mask

        return img


if __name__ == '__main__':
    img = cv2.imread("test.jpg")
    img = np.asarray(img, dtype='float32')

    img = np.transpose(img, [2, 0, 1])
    img /= 255
    img = RandomMask()(img, mag=2)
    img = np.transpose(img, [1, 2, 0])
    img = (img * 255).astype('uint8')

    # print(img)

    cv2.imwrite('distort.jpg', img)
