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
Load mini-imagenet-dataset from hard drive.
"""
import paddle
from paddle.io import Dataset
import cv2
import os
import numpy as np
import collections

import sys
sys.path.append("../..")
from paddle.vision import transforms as T


class MiniImageNet(Dataset):
    """
    Implementation of Mini-ImageNet dataset.

    Args:
        mode(str): 'train', 'test', or 'val' mode. Default 'test'.
        transform: function to call to preprocess images.

    Returns:
        Dataset: Mini-Imagenet Dataset.

    Examples:
        .. code-block:: python

            from miniimagenet import MINIIMAGENET
            test_dataset = MINIIMAGENET(mode='test')
            test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=False)
            for id, data in enumerate(test_loader):
                image = data[0]
                label = data[1]
    """
    def __init__(self,
                 mode='test',
                 transform=None):
        self.transform = transform
        parent_folder = './mini-imagenet/'
        train_data_path = 'images'
        test_data_path = 'images'
        val_data_path = 'images'
        train_label_path = 'train.csv'
        test_label_path = 'test.csv'
        val_label_path = 'val.csv'
        supported_mode = ('train', 'val', 'test')

        assert mode in supported_mode
        if mode == 'train':
            image_path = os.path.join(parent_folder, train_data_path)
            label_path = os.path.join(parent_folder, train_label_path)
        elif mode == 'val':
            image_path = os.path.join(parent_folder, val_data_path)
            label_path = os.path.join(parent_folder, val_label_path)
        elif mode == 'test':
            image_path = os.path.join(parent_folder, test_data_path)
            label_path = os.path.join(parent_folder, test_label_path)
        else:
            exit(0)

        self.labelinfo = self._load_labelinfo(image_path, label_path)

    def _load_labelinfo(self, image_path, label_path):
        image_filepaths = os.listdir(image_path)
        with open(label_path, 'r') as file:
            label_info_list = file.readlines()

        labelinfo = []
        for inedx, info in enumerate(label_info_list):
            if inedx == 0:
                continue
            splited = info.split(',')
            image_filename = splited[0]
            image_label = splited[1].strip()

            if image_filename in image_filepaths:
                image_filepath = os.path.join(image_path, image_filename)
                labelinfo.append((image_filepath, image_label))

        return labelinfo

    def __getitem__(self, idx):
        image_filepath, label = self.labelinfo[idx]
        image = cv2.imread(image_filepath)
        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = image
        return transformed_image, label

    def __len__(self):
        return len(self.labelinfo)


def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Set the testing set
    transform_train = T.Compose([T.Resize((224, 224)),
                                 T.RandomHorizontalFlip(0.5),
                                 T.RandomVerticalFlip(0.5),
                                 T.Transpose(),
                                 T.Normalize(
                                     mean=[0, 0, 0],
                                     std=[255, 255, 255]),
                                 T.Normalize(mean, std, data_format='CHW')
                                 ])
    transform_eval = T.Compose([T.Resize((224, 224)),
                                 T.Transpose(),
                                 T.Normalize(
                                     mean=[0, 0, 0],
                                     std=[255, 255, 255]),
                                 T.Normalize(mean, std, data_format='CHW')
                                ])

    # Set the classification network
    model1 = paddle.vision.models.resnet50(pretrained=True)
    model2 = paddle.vision.models.vgg16(pretrained=True)
    model7 = paddle.vision.models.mobilenet_v1(pretrained=True)

    train_set = MiniImageNet(mode='train', transform=transform_train)
    test_set = MiniImageNet(mode='test', transform=transform_eval)
    train_loader = paddle.io.DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = paddle.io.DataLoader(test_set, batch_size=16, shuffle=True)

    train_label_dict = collections.defaultdict(int)
    test_label_dict = collections.defaultdict(int)
    for index, data in enumerate(train_loader):
        images = data[0]
        labels = data[1]
        for label in labels:
            train_label_dict[label] += 1
        # predict1 = model1(images)
        # predict2 = model2(images)
        # predict7 = model7(images)
        # label1 = np.argmax(predict1, axis=1)
        # label2 = np.argmax(predict2, axis=1)
        # label7 = np.argmax(predict7, axis=1)
        # print(label1)
        # print(label2)
        # print(label7)
        #

        # cv2.imshow('tmp', (images.numpy() * 255).astype(int))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    for index, data in enumerate(test_loader):
        images = data[0]
        labels = data[1]
        for label in labels:
            test_label_dict[label] += 1

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
