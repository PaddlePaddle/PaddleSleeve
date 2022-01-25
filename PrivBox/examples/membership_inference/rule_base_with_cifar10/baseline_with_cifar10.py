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
This module provides an example of baseline membership inference attack on Cifar10.
"""

from __future__ import print_function

import os

import argparse
import numpy
import numpy as np

import paddle
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F

from privbox.inference.membership_inference import BaselineMembershipInferenceAttack
from privbox.metrics import MSE, Accuracy, AUC, Precision, Recall


def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("BaselineMembershipInferenceAttack")
    parser.add_argument("--batch_size",
                        type=int, default=128,
                        help="The batch size of normal training.")
    parser.add_argument("--train_epoch",
                        type=int, default=10,
                        help="The epoch of training.")
    parser.add_argument("--train_lr",
                        type=float, default=0.0002,
                        help="The learning rate of training.")
    args = parser.parse_args()
    return args


class ResNet(paddle.nn.Layer):
    """
    Use ResNet18 for Cifar10
    """
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        kwargs = {"num_classes": num_classes}
        self.res_net = paddle.vision.resnet18(**kwargs)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.res_net(x)
        return y


def get_data():
    """
    get train dataset and test dataset for cifar10
    """
    transform = Compose([paddle.vision.Resize(32),
                    Normalize(mean=[127.5], std=[127.5], data_format='CHW'),
                    paddle.vision.transforms.Transpose()])
    train_data = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
    l = len(train_data)
    return paddle.io.random_split(train_data, [l // 2, l - l // 2])


def get_all_labels(data_list):
    """
    get labels from multiple dataset
    """
    labels = []
    for dataset in data_list:
        for data in dataset:
            labels.append(data[1])

    return paddle.to_tensor(labels)


def train_and_attack(args):
    """
    The training procedure that starts from training target model,
    then launchs baseline membership inference attack

    Args:
        args(ArgumentParser): the execution parameters.
    """
    mem_data, non_mem_data = get_data()

    num_classes = 10

    model = paddle.Model(ResNet(num_classes))

    # train model
    model.prepare(paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.train_lr),
                  paddle.nn.CrossEntropyLoss(),
                   [paddle.metric.Accuracy()])
    print("training target model ...")
    model.fit(mem_data, non_mem_data, epochs=args.train_epoch, verbose=1, batch_size=args.batch_size)

    # get predict result
    mem_pred = model.predict(mem_data, batch_size=args.batch_size, stack_outputs=True)
    non_mem_pred = model.predict(non_mem_data, batch_size=args.batch_size, stack_outputs=True)

    mem_pred = paddle.argmax(paddle.to_tensor(mem_pred[0]), axis=-1)
    non_mem_pred = paddle.argmax(paddle.to_tensor(non_mem_pred[0]), axis=-1)

    input_data = paddle.concat([mem_pred, non_mem_pred], axis=0)
    input_label = get_all_labels([mem_data, non_mem_data])

    # membership attack
    attack = BaselineMembershipInferenceAttack()
    result = attack.infer([input_data, input_label])

    # evaluate
    mem_label = paddle.ones(mem_pred.shape)
    non_mem_label = paddle.zeros(non_mem_pred.shape)
    expected = paddle.concat([mem_label, non_mem_label], axis=0)
    eval_res = attack.evaluate(result, expected, metric_list=[Accuracy(False, 2), AUC(False), Precision(), Recall()])

    print("""Evaluate result of baseline membership attack on cifar10 is: acc: {},
          auc: {}, precision: {}， recall: {}""".format(eval_res[0],
          eval_res[1], eval_res[2], eval_res[3]))

    print("Attack finish")


if __name__ == "__main__":
    arguments = parse_args()
    print("args: ", arguments)
    train_and_attack(arguments)
