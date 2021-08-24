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
Load mini-imagenet-dataset from the downloaded .pkl files
"""

from __future__ import print_function

import paddle
from paddle.io import Dataset
import numpy as np
from six.moves import cPickle as pickle

__all__ = []

class MINIIMAGENET(Dataset):
    """
    Implementation of `MINI-IMAGENET <https://github.com/yaoyao-liu/mini-imagenet-tools>`_ dataset

    Args:
        dataset_path(str): path to .pkl image file, can be set None if, default data path: input/mini-imagenet-cache-test.pkl
        label_path(str): path to .txt label file, can be set None if, default label path: input/mini_imagenet_test_labels.txt
        mode(str): 'train', 'test', or 'val' mode. Default 'test'.

    Returns:
        Dataset: MINIIMAGENET Dataset.

    Examples:

        .. code-block:: python

            from miniimagenet import MINIIMAGENET

            test_dataset = MINIIMAGENET(mode='test')
            test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=False)
            for id, data in enumerate(test_loader):
                image = data[0]
                label = data[1]

    """
    NAME = 'miniimagenet'
    TRAIN_DATA_PATH = 'input/mini-imagenet-cache-train.pkl'
    TEST_DATA_PATH = 'input/mini-imagenet-cache-test.pkl'
    VAL_DATA_PATH = 'input/mini-imagenet-cache-val.pkl'
    TRAIN_LABEL_PATH = 'input/mini_imagenet_train_labels.txt'
    TEST_LABEL_PATH = 'input/mini_imagenet_test_labels.txt'
    VAL_LABEL_PATH = 'input/mini_imagenet_val_labels.txt'

    def __init__(self,
                 dataset_path=None,
                 label_path=None,
                 mode='test',
                 transform=None,
                 backend=None):
        assert mode.lower() in ['train', 'test', 'val'], \
            "mode should be 'train', 'test', or 'val', but got {}".format(mode)

        self.mode = mode.lower()

        if dataset_path is None:
            if self.mode == 'train':
                dataset_path = self.TRAIN_DATA_PATH
            elif self.mode == 'test':
                dataset_path = self.TEST_DATA_PATH
            else:
                dataset_path = self.VAL_DATA_PATH

        if label_path is None:
            if self.mode == 'train':
                label_path = self.TRAIN_LABEL_PATH
            elif self.mode == 'test':
                label_path = self.TEST_LABEL_PATH
            else:
                label_path = self.VAL_LABEL_PATH
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.transform = transform

        # Read dataset into memory
        self._load_data()

        self.dtype = paddle.get_default_dtype()

    def _load_data(self):
        '''
        Open .pkl file and store it to self.data
        :return: None.
        '''
        self.data = []
        # Process .pkl file
        with open(self.dataset_path, mode='rb') as f:
            dict = pickle.load(f)
            image_data = dict['image_data']
            label_dict = dict['class_dict']
        # find the corresponding labels
        with open(self.label_path) as info:
            mini_imagenet_labels = eval(info.read())

        new_label_dict = {}
        for k, v in label_dict.items():
            for i in v:
                new_label_dict[i] = mini_imagenet_labels[k]
                self.data.append((image_data[i], new_label_dict[i]))

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # Transpose image shape from CHW to HWC
        image = image.transpose([2, 0, 1])
        # Scale image range from [0, 255] to [0, 1]
        image = image / 255.0
        # Normalise image
        image = self.transform(image)
        return image.astype(self.dtype), np.array(label).astype('int64')

    def __len__(self):
        return len(self.data)

'''
content of mini_imagenet_test_labels.txt
{'n03544143': 604,
'n03127925': 519,
'n03146219': 524,
'n04418357': 854,
'n02110063': 249,
'n07613480': 927,
'n02116738': 275,
'n03775546': 659,
'n02443484': 359,
'n01930112': 111,
'n02099601': 207,
'n02129165': 291,
'n02871525': 454,
'n03272010': 546,
'n04149813': 781,
'n02219486': 310,
'n04522168': 883,
'n02110341': 251,
'n04146614': 779,
'n01981276': 121}
'''