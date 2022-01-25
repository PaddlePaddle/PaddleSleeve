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
This module provides an example of Knockoff attack on MNIST.
"""

from __future__ import print_function

import os

import argparse
import numpy
import numpy as np
import pdb

import paddle
from PIL import Image
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F
from paddle.vision.models.resnet import BasicBlock
from privbox.extraction.knockoff_nets import KnockoffExtractionAttack
from privbox.metrics import MSE, Accuracy


def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("Knockoff")
    parser.add_argument("--batch_size",
                        type=int, default=128,
                        help="The batch size of training and predict.")
    parser.add_argument("--epochs",
                        type=int, default=2,
                        help="The iterations of training for victim and adversary.")
    parser.add_argument("--learning_rate",
                        type=float, default=0.01,
                        help="The learning rate of training for victim and adversary.")

    parser.add_argument("--num_queries",
                        type=int, default=2000,
                        help="The number of queries allowed for adversary.")

    parser.add_argument("--knockoff_net",
                        type=str, default="linear",
                        choices=["linear", "resnet"],
                        help="The newwork for knockoff model, can be chosen from 'linear' and 'resnet'.")

    parser.add_argument("--knockoff_dataset",
                        type=str, default="mnist",
                        choices=["mnist", "fmnist"],
                        help="The dataset for training knockoff model, "
                        "can be chosen from 'mnist' (100% labels overlap) and 'fmnist' (0% labels overlap).")

    parser.add_argument("--policy",
                        type=str, default="random",
                        choices=["random", "adaptive"],
                        help="The policy for data sampling, can be chosen from 'random' and 'adaptive'.")

    parser.add_argument("--reward",
                        type=str, default="all",
                        choices=["certainty", "diversity", "loss", "all"],
                        help="The reward strategy for adaptive policy.")

    args = parser.parse_args()
    return args


class MNISTResNet(paddle.vision.ResNet):
    """
    ResNet18 for MNIST
    """
    def __init__(self):
        """
        Init function
        Origin ResNet's channel is 3, where MNIST picture channel is 1,
        therefore we need to change ResNet layer params
        """
        super(MNISTResNet, self).__init__(BasicBlock, 18, num_classes=10)
        self.conv1 = paddle.nn.Conv2D(1, 64, kernel_size=(7, 7),
                                      stride=(2, 2), padding=(3, 3))


class LinearNet(paddle.nn.Layer):
    """
    Define a simple Linear Network for MNIST
    """
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear = paddle.nn.Linear(28 * 28, 10)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.linear(paddle.reshape(x, [-1, 28 * 28]))
        y = y.reshape([-1, 10])
        return y


class QueryFunctor(object):
    """
    A callable functor for adversary to use data to query label.
    Output must be predict posterior probability
    """
    def __init__(self, model):
        self.model= model

    def __call__(self, x):
        return self.model.network(x)


def train_and_attack(args):
    """
    The training procedure that starts from training a image classifier,
    Then an attack is constructed to extract the training model

    Args:
        args(ArgumentParser): the execution parameters.
    """

    if args.knockoff_dataset == "fmnist" and args.policy == "adaptive":
        raise ValueError("Can't set policy as 'adaptive' for 'fmnist' dataset.")
    
    # load mnist data and split for test data and attacker's query data
    transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_set = paddle.vision.datasets.MNIST(mode='test', transform=transform)

    train_knock_len = int(4.0 / 5 * len(test_set))

    if args.num_queries < train_knock_len:
        train_knock_len = args.num_queries


    lengths = [train_knock_len, len(test_set) - train_knock_len]

    [train_knockoff_dataset, test_dataset] = paddle.io.random_split(test_set, lengths)

    if args.knockoff_dataset == "fmnist":
        fmnist_dataset = paddle.vision.datasets.FashionMNIST(mode='train', transform=transform)
        train_knockoff_dataset = fmnist_dataset

    # Define and train Victim's model
    victim_net = LinearNet()

    v_model = paddle.Model(victim_net)

    v_model.prepare(optimizer=paddle.optimizer.Adam(
                    parameters=v_model.parameters(),
                            learning_rate=args.learning_rate),
                    loss=paddle.nn.CrossEntropyLoss(),
                    metrics=paddle.metric.Accuracy())

    v_model.fit(train_loader, batch_size=args.batch_size, epochs=args.epochs)

    # Define Knockoff extraction attack
    knockoff_net = LinearNet() if args.knockoff_net == "linear" else MNISTResNet()
    attack = KnockoffExtractionAttack(QueryFunctor(v_model), knockoff_net)

    # set attack params
    params = {"policy": args.policy, "has_label": True, "reward": args.reward,
              "num_labels": 10, "num_queries": args.num_queries,
              "knockoff_batch_size": args.batch_size, "knockoff_epochs": args.epochs,
              "knockoff_lr": args.learning_rate}

    attack.set_params(**params)

    # extract model
    knockoff_net = attack.extract(train_knockoff_dataset)

    # evaluate attack
    kwargs_dataset = {"test_dataset": test_dataset}
    ret = attack.evaluate(victim_net, knockoff_net, [Accuracy()], **kwargs_dataset)

    print("Victim model's evaluate result: ", ret[0])
    print("Knockoff model's evaluate result: ", ret[1])


if __name__ == "__main__":
    arguments = parse_args()
    print("input args: \n", arguments)
    train_and_attack(arguments)
    print("attack finish.")
