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
import numpy as np
import cv2
import os
import collections

import sys
sys.path.append("../..")
from paddle.vision import transforms as T


class MiniImageNet1(Dataset):
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
        # TODO: make it more general
        parent_folder = '../dataset/mini-imagenet1/mini-imagenet'
        supported_mode = ('train', 'val', 'test')
        assert mode in supported_mode

        subclass_folders = os.listdir(parent_folder)

        self.image_2_label = []
        for i, folder in enumerate(subclass_folders):
            image_folder = os.path.join(parent_folder, folder, mode)
            image_filenames = os.listdir(image_folder)
            for image_filename in image_filenames:
                image_path = os.path.join(image_folder, image_filename)
                self.image_2_label.append((image_path, i))

    def __getitem__(self, idx):
        image_filepath, label = self.image_2_label[idx]
        image = cv2.imread(image_filepath)
        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = image
        return transformed_image, label

    def __len__(self):
        return len(self.image_2_label)


def main():
    """
    Main for running a tutorial for ImageNet.
    Returns:
        None
    """
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

    train_set = MiniImageNet1(mode='train', transform=transform_train)
    test_set = MiniImageNet1(mode='test', transform=transform_eval)
    train_loader = paddle.io.DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = paddle.io.DataLoader(test_set, batch_size=16, shuffle=True)

    train_label_dict = collections.defaultdict(int)
    test_label_dict = collections.defaultdict(int)
    for index, data in enumerate(train_loader):
        images = data[0]
        labels = data[1]
        for label in labels:
            train_label_dict[label] += 1
        predict1 = model1(images)
        predict2 = model2(images)
        predict7 = model7(images)
        label1 = np.argmax(predict1, axis=1)
        label2 = np.argmax(predict2, axis=1)
        label7 = np.argmax(predict7, axis=1)
        print(label1)
        print(label2)
        print(label7)

        import pdb
        pdb.set_trace()
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
