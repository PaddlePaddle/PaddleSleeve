#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
Implementation AT&T face dataset
"""
from __future__ import print_function
from six.moves import range
from PIL import Image, ImageOps

import zipfile
import numpy as np
import argparse
import struct
import os
import paddle
import random

from paddle.dataset.common import _check_exists_and_download

class ATTFace(paddle.io.Dataset):
    """
    Implementation of `AT&T face`_ dataset
    Args:
        image_path(str): path to image file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/att_face
        label_path(str): path to label file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/att_face
        mode(str): 'train' or 'test' mode. Default 'train'.
        download(bool): download dataset automatically if
            :attr:`image_path` :attr:`label_path` is not set. Default True
            
    Returns:
        Dataset: AT&T Face Dataset.
    """
    NAME = 'att_face'
    URL = """http://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/10-Eigenimages/Eigenfaces_Versus_Fisherfaces/att_faces.zip"""
    MD5 = '84dbcc1fbeea1f9f7850d3631d3ff599'

    def __init__(self,
                 image_path=None,
                 label_path=None,
                 mode='train',
                 transform=None,
                 download=True):
        assert mode.lower() in ['train', 'test'], \
                "mode should be 'train' or 'test', but got {}".format(mode)

        self.mode = mode.lower()
        self.image_path = image_path
        if self.image_path is None:
            assert download, "image_path is not set and downloading automatically is disabled"
            
            self.image_path = _check_exists_and_download(
                image_path, self.URL, self.MD5, self.NAME, download)

        self.transform = transform

        # read dataset into memory
        self._parse_dataset()

        self.dtype = paddle.get_default_dtype()

    def _parse_dataset(self, buffer_size=400):
        # since AT&T data is smalll, we can read all data into memory
        self.images = []
        self.labels = []
        with zipfile.ZipFile(self.image_path, 'r') as image_file:
            data_dir = os.path.dirname(self.image_path)
            for file in image_file.namelist():
                image_file.extract(file, data_dir)
            for i in range(1, 41):
                dir = data_dir + "/att_faces" + "/s" + str(i) + "/"
                for j in range(1, 11):
                    file = dir + str(j) + ".pgm"
                    img = Image.open(file, "r")
                    self.images.append(np.array(img))
                    self.labels.append(
                    np.array([i - 1]).astype('int64'))
                
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = np.reshape(image, [112, 92])

        if self.transform is not None:
            image = Image.fromarray(image.astype('uint8'), mode='L')
            image = self.transform(image)
            image = np.array(image)

        return image.astype(self.dtype), label.astype('int64')

    def __len__(self):
        return len(self.labels)
