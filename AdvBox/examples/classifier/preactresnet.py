# Pre-activation ResNet in paddlepaddle.

from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from paddle.vision import transforms as T
import numpy as np

__all__ = ['preactresnet18']


class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')


MEAN = None
STD = None
transform_train = T.Compose([T.RandomCrop(size=32, padding=4),
                             T.RandomHorizontalFlip(0.5),
                             ToArray()])
transform_eval = T.Compose([T.Resize(size=32),
                            ToArray()])


class PreActBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(PreActBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self.bn1 = norm_layer(inplanes)
        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())
        self.relu = nn.ReLU()

        self.bn2 = norm_layer(planes)
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())

        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())
            )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(PreActBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        self.bn1 = norm_layer(inplanes)
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())
        self.relu = nn.ReLU()
        self.bn2 = norm_layer(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())
        self.bn3 = norm_layer(planes)
        self.conv3 = nn.Conv2D(planes, self.expansion * planes, kernel_size=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())

        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())
            )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.
    Examples:
        .. code-block:: python
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock
            resnet50 = ResNet(BottleneckBlock, 50)
            resnet18 = ResNet(BasicBlock, 18)
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64

        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal())
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.relu = nn.ReLU()
        self.bn = self._norm_layer(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, norm_layer))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x


def preactresnet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


def preactresnet34(num_classes=10):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)


def preactresnet50(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)


def preactresnet101(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes)


def preactresnet152(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == '__main__':
    net = preactresnet18()
    x = paddle.randn([1, 3, 32, 32], dtype='float32')
    y = net(x)