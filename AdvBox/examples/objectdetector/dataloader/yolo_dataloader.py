# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Load image from hard drive(folder). For model testing.
"""

from paddle.io import Dataset
import os
import cv2
import paddle


class PPDETFolderReader(Dataset):
    def __init__(self,
                 folder_path='',
                 transform=None):
        self.folder_path = folder_path
        self.transform = transform

        image_folder = self.folder_path
        image_filenames = os.listdir(image_folder)
        self.imagepath_2_filename = []
        for image_filename in image_filenames:
            image_path = os.path.join(image_folder, image_filename)
            self.imagepath_2_filename.append((image_path, image_filename))

    def __getitem__(self, idx):
        image_filepath, image_filename = self.imagepath_2_filename[idx]
        reading_success = False
        while not reading_success:
            try:
                image = cv2.imread(image_filepath)
                if self.transform is not None:
                    # TODO: fix bug here.
                    transformed_image = self.transform(image)
                else:
                    transformed_image = image
                reading_success = True
            except Exception as e:
                print(e)

        data = {}
        transformed_image = paddle.unsqueeze(transformed_image, axis=0)
        data['im_id'] = paddle.to_tensor([[idx]], place=paddle.fluid.CUDAPinnedPlace())
        data['curr_iter'] = paddle.to_tensor(idx, place=paddle.fluid.CUDAPinnedPlace())
        data['image'] = transformed_image
        data['im_shape'] = paddle.to_tensor([[320, 320]], dtype='float32')
        data['scale_factor'] = paddle.to_tensor([[0.5, 0.5]], dtype='float32')

        return data

    def __len__(self):
        return len(self.imagepath_2_filename)
