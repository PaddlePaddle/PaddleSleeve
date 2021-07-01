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
自定义paddle模型用于：对抗生成 & 对抗训练
"""
from __future__ import division
import paddle
import paddle.nn.functional as F
from paddle.vision import transforms as T

__all__ = [
    "MyNet", "TowerNet"
]

MEAN = [0.5095, 0.5463, 0.5741]
STD = [0.2606, 0.2604, 0.2922]
transform_train = T.Compose([T.Resize((256, 256)),
                             T.RandomHorizontalFlip(0.5),
                             T.RandomVerticalFlip(0.5),
                             T.Transpose(),
                             T.Normalize(
                                 mean=[0, 0, 0],
                                 std=[255, 255, 255]),
                             # output[channel] = (input[channel] - mean[channel]) / std[channel]
                             T.Normalize(mean=MEAN,
                                         std=STD)
                             ])
transform_eval = T.Compose([T.Resize((256, 256)),
                            T.Transpose(),
                            T.Normalize(
                                mean=[0, 0, 0],
                                std=[255, 255, 255]),
                            # output[channel] = (input[channel] - mean[channel]) / std[channel]
                            T.Normalize(mean=MEAN,
                                        std=STD)
                            ])


class Inception(paddle.nn.Layer):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 路线1，卷积核1x1
        self.route1x1_1 = paddle.nn.Conv2D(in_channels, c1, kernel_size=1)
        # 路线2，卷积层1x1、卷积层3x3
        self.route1x1_2 = paddle.nn.Conv2D(in_channels, c2[0], kernel_size=1)
        self.route3x3_2 = paddle.nn.Conv2D(c2[0], c2[1], kernel_size=3, padding=1)
        # 路线3，卷积层1x1、卷积层5x5
        self.route1x1_3 = paddle.nn.Conv2D(in_channels, c3[0], kernel_size=1)
        self.route5x5_3 = paddle.nn.Conv2D(c3[0], c3[1], kernel_size=5, padding=2)
        # 路线4，池化层3x3、卷积层1x1
        self.route3x3_4 = paddle.nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.route1x1_4 = paddle.nn.Conv2D(in_channels, c4, kernel_size=1)

    def forward(self, x):
        route1 = F.relu(self.route1x1_1(x))
        route2 = F.relu(self.route3x3_2(F.relu(self.route1x1_2(x))))
        route3 = F.relu(self.route5x5_3(F.relu(self.route1x1_3(x))))
        route4 = F.relu(self.route1x1_4(self.route3x3_4(x)))
        out = [route1, route2, route3, route4]
        # 在通道维度(axis=1)上进行连接
        return paddle.concat(out, axis=1)


def BasicConv2d(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = paddle.nn.Sequential(
                paddle.nn.Conv2D(in_channels, out_channels, kernel, stride, padding),
                paddle.nn.BatchNorm2D(out_channels, epsilon=1e-3),
                paddle.nn.ReLU())
    return layer


class Residual(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel, stride=1, wide_scale=1):
        super(Residual, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channel, out_channel, kernel_size=1, stride=stride)
        self.b5 = paddle.nn.Sequential(Inception(256 * wide_scale,
                                                 64 * wide_scale,
                                                 (64 * wide_scale, 128 * wide_scale),
                                                 (16 * wide_scale, 32 * wide_scale),
                                                 32 * wide_scale))

    def forward(self, x):
        y = self.b5(x)
        x = self.conv1(x)
        # Core spirit!!!
        out = F.relu(y + x)
        return out


# build up the Tower
class TowerNet(paddle.nn.Layer):
    def __init__(self, in_channel, num_classes, wide_scale=1):
        super(TowerNet, self).__init__()
        self.b1 = paddle.nn.Sequential(
                    BasicConv2d(in_channel, out_channels=64 * wide_scale, kernel=3, stride=2, padding=1),
                    paddle.nn.MaxPool2D(2, 2))
        self.b2 = paddle.nn.Sequential(
                    BasicConv2d(64 * wide_scale, 128 * wide_scale, kernel=3, padding=1),
                    paddle.nn.MaxPool2D(2, 2))
        self.b3 = paddle.nn.Sequential(
                    BasicConv2d(128 * wide_scale, 256 * wide_scale, kernel=3, padding=1),
                    paddle.nn.MaxPool2D(2, 2))
        self.b4 = paddle.nn.Sequential(
                    BasicConv2d(256 * wide_scale, 256 * wide_scale, kernel=3, padding=1),
                    paddle.nn.MaxPool2D(2, 2))
        self.b5 = paddle.nn.Sequential(
                    Residual(256 * wide_scale, 256 * wide_scale, wide_scale=wide_scale),
                    paddle.nn.MaxPool2D(2, 2),
                    Residual(256 * wide_scale, 256 * wide_scale, wide_scale=wide_scale),
                    paddle.nn.MaxPool2D(2, 2),
                    Residual(256 * wide_scale, 256 * wide_scale, wide_scale=wide_scale))
        self.AvgPool2D = paddle.nn.AvgPool2D(2)
        self.flatten = paddle.nn.Flatten()
        self.b6 = paddle.nn.Linear(256 * wide_scale, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.AvgPool2D(x)
        x = self.flatten(x)
        x = self.b6(x)
        return x


# an arbitrary neural network
class MyNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(MyNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
