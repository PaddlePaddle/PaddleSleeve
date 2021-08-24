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
self defined paddle cnn model
"""
import paddle
import paddle.nn.functional as F


# TODO: move it to classifier folder.
class CNNModel(paddle.nn.Layer):
    """
    paddle CNN model
    """

    def __init__(self):
        """
        init
        """
        super(CNNModel, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=800, out_features=50)
        self.linear2 = paddle.nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

