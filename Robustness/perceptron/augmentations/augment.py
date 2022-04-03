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

# Compose operators to enable sequence augmentations
import os.path
import traceback
import sys
sys.path.append("../..")

import cv2
import numpy as np
from perceptron.augmentations import operators


def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png')):
    return f.lower().endswith(extensions)


def _make_dataset(dir):
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ('{} should be a dir'.format(dir))
    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if _is_valid_file(path):
                images.append(path)
    return images


class SerialAugment(object):
    def __init__(self,
                 transforms=[],
                 format='CHW',
                 bound=(0, 1),
                 input_path=None,
                 output_path=None):
        super(SerialAugment, self).__init__()
        self._input_path = input_path
        self._output_path = output_path
        self._format = format
        self._bound = bound
        self._transforms = transforms
        self._ops = []
        self.img_count = 0
        for t in self._transforms:
            for k, v in t.items():
                op_cls = getattr(operators, k)
                v['format'] = self._format
                v['bound'] = self._bound
                f = op_cls(**v)
                self._ops.append(f)

    def apply(self, img, **kwargs):
        for f in self._ops:
            try:
                img = f(img, **kwargs)
            except Exception as e:
                stack_info = traceback.format_exc()
                raise e
        return img

    def __call__(self, images=None, **kwargs):
        if images is not None:
            images = self.apply(images, **kwargs)
            if self._output_path is not None:
                self.save_image(images)
        elif self._input_path is not None:
            img_path = self.parse_images(self._input_path)
            images = self.load_images(img_path)
            images = self.apply(images)

            if self._output_path is not None:
                self.save_image(images, name=img_path)
        else:
            print('Please provide the images to be augmented')
            exit(-1)

        return images

    def set_image(self, path=None):
        self._input_path = path

    def set_out_path(self, path=None):
        self._output_path = path

    def load_images(self, images):
        rec = []
        for image in images:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self._format == 'CHW':
                img = np.transpose(img, [2, 0, 1])
            if self._bound == (0, 1):
                img = img / 255
            rec.append(img)
        return rec

    def parse_images(self, image_dir=None):
        if not isinstance(image_dir, list):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def save_image(self, images=None, name=None):
        if not isinstance(images, list):
            images = [images]

        assert name is None or len(images) <= len(name), 'Not enough image names'
        out_dir = self._output_path

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for i, image in enumerate(images):
            if self._format == 'CHW':
                image = np.transpose(image, [1, 2, 0])
            if self._bound == (0, 1):
                image = (image * 255).astype('uint8')
            if name is not None:
                img_name = os.path.basename(name[i])
            else:
                img_name = '{:012d}.jpg'.format(self.img_count)
            img_name = os.path.join(out_dir, img_name)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_name, image)
            self.img_count += 1

