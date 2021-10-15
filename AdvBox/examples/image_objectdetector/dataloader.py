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
Load image from hard drive(folder).
"""
# TODO: finish a carmera intake folder reading dataloader
import paddle
from paddle.io import Dataset
import numpy as np
import cv2
import os
import collections

import sys
sys.path.append("../..")


class FolderReader(Dataset):
    def __init__(self,
                 mode='test',
                 transform=None):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.image_2_label)
