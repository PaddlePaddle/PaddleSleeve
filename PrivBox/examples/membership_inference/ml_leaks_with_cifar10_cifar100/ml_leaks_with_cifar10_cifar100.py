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
This module provides an example of ML_Leaks membership inference attack.
"""

from __future__ import print_function

import os

import argparse
import numpy
import numpy as np
import time

import paddle
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F

from privbox.inference.membership_inference import MLLeaksMembershipInferenceAttack
from privbox.metrics import MSE, Accuracy, AUC, Precision, Recall


def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("ML_Leaks")
    parser.add_argument("--batch_size",
                        type=int, default=128,
                        help="The batch size of normal training.")
    parser.add_argument("--target_epoch",
                        type=int, default=10,
                        help="The iterations of target model training.")
    parser.add_argument("--shadow_epoch",
                        type=int, default=10,
                        help="The iterations of shadow model training.")
    parser.add_argument("--classifier_epoch",
                        type=int, default=10,
                        help="The iterations of classifier training.")
    parser.add_argument("--target_lr",
                        type=float, default=0.0002,
                        help="The learning rate of target model training.")
    parser.add_argument("--shadow_lr",
                        type=float, default=0.0002,
                        help="The learning rate of shadow model training.")
    parser.add_argument("--classifier_lr",
                        type=float, default=0.0002,
                        help="The learning rate of classifier training.")
    parser.add_argument("--topk",
                        type=int, default=10,
                        help="The top k predict results that used for training classifier.")
    parser.add_argument("--shadow_dataset",
                        type=str, choices=["cifar10", "cifar100"], default="cifar10",
                        help="shadow dataset")
    parser.add_argument("--target_dataset",
                        type=str, choices=["cifar10", "cifar100"], default="cifar10",
                        help="target dataset")
    parser.add_argument("--shadow_model", type=str, choices=['resnet18', 'resnet34'],
                        default='resnet18',
                        help="using what shadow model resnet18 or resnet34")
    args = parser.parse_args()
    return args


class ResNet34(paddle.nn.Layer):
    """
    Define ResNet34
    """
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        kwargs = {"num_classes": num_classes}
        self.res_net = paddle.vision.resnet34(**kwargs)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.res_net(x)
        return y


class ResNet18(paddle.nn.Layer):
    """
    Define ResNet18
    """
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        kwargs = {"num_classes": num_classes}
        self.res_net = paddle.vision.resnet18(**kwargs)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.res_net(x)
        return y


def split_dataset(dataset, target_prob=0.5, batch_size=32):
    """
    split dataset to target dataset and shadow dataset
    """
    total_len = len(dataset)
    target_len = int(total_len * target_prob)
    shadow_len = total_len - target_len

    sets = paddle.io.random_split(dataset, [target_len, shadow_len])
    loader0 = paddle.io.DataLoader(sets[0], shuffle=True, batch_size=batch_size)
    loader1 = paddle.io.DataLoader(sets[1], shuffle=True, batch_size=batch_size)
    return loader0, loader1


def train_and_attack(args):
    """
    The training procedure that starts from training target model,
    then launchs ML_Leaks membership inference attack

    Args:
        args(ArgumentParser): the execution parameters.
    """
    
    transform = Compose([paddle.vision.Resize((32, 32)),
                    Normalize(mean=[127.5], std=[127.5], data_format='CHW'),
                    paddle.vision.transforms.Transpose()])
    used_dataset = (args.shadow_dataset, args.target_dataset)

    # load choose datasets
    if "cifar10" in used_dataset:
        cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
        target_cifar10_train, shadow_cifar10_train = split_dataset(cifar10_train, batch_size=args.batch_size)
        cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform)
        target_cifar10_test, shadow_cifar10_test = split_dataset(cifar10_test, batch_size=args.batch_size)

    if "cifar100" in used_dataset:
        cifar100_train = paddle.vision.datasets.Cifar100(mode='train', transform=transform)
        target_cifar100_train, shadow_cifar100_train = split_dataset(cifar100_train, batch_size=args.batch_size)
        cifar100_test = paddle.vision.datasets.Cifar100(mode='test', transform=transform)
        target_cifar100_test, shadow_cifar100_test = split_dataset(cifar100_test, batch_size=args.batch_size)

    def get_dataset(name, mode='shadow'):
        """
        get dataset from name
        """
        train_data = None
        test_data = None
        num_classes = None
        if name == "cifar10" and mode == 'shadow':
            train_data = shadow_cifar10_train
            test_data = shadow_cifar10_test
            num_classes = 10
        elif name == "cifar100" and mode == 'shadow':
            train_data = shadow_cifar100_train
            test_data = shadow_cifar100_test
            num_classes = 100
        elif name == "cifar10" and mode == 'target':
            train_data = target_cifar10_train
            test_data = target_cifar10_test
            num_classes = 10
        elif name == "cifar100" and mode == 'target':
            train_data = target_cifar100_train
            test_data = target_cifar100_test
            num_classes = 100
        else:
            raise ValueError("no data set name {}".format(name))
        return train_data, test_data, num_classes

    # get shadow and target data
    shadow_train_data, shadow_test_data, shadow_num_classes = get_dataset(args.shadow_dataset)

    target_train_data, target_test_data, target_num_classes = get_dataset(args.target_dataset, mode='target')

    # define target model
    target_model = ResNet18(num_classes=target_num_classes)
    target_model = paddle.Model(target_model)
    print("Begin training target model")

    train_model(target_model, target_train_data, target_test_data,
                epoch=args.target_epoch, learning_rate=args.target_lr, batch_size=args.batch_size)

    # define shadow model
    shadow_model = None
    if args.shadow_model == 'resnet18':
        print("Shadow model is set to ResNet18")
        shadow_model = paddle.Model(ResNet18(shadow_num_classes))
    else:
        print("Shadow model is set to ResNet34")
        shadow_model = paddle.Model(ResNet34(shadow_num_classes))
    print("Begin training shadow model")

    # define attack
    attack = MLLeaksMembershipInferenceAttack(shadow_model, shadow_dataset=[shadow_train_data, shadow_test_data])

    attack_params = {"batch_size": args.batch_size, "shadow_epoch": args.shadow_epoch,
                     "classifier_epoch": args.classifier_epoch, "topk": args.topk,
                     "shadow_lr":args.shadow_lr, "classifier_lr": args.classifier_lr}

    # set params
    attack.set_params(**attack_params)


    print("Infer target dataset")
    mem_pred = target_model.predict(target_train_data, batch_size=args.batch_size, stack_outputs=True)
    non_mem_pred = target_model.predict(target_test_data, batch_size=args.batch_size, stack_outputs=True)

    mem_pred = paddle.to_tensor(mem_pred[0])
    non_mem_pred = paddle.to_tensor(non_mem_pred[0])

    data = paddle.concat([mem_pred, non_mem_pred])

    result = attack.infer(data)

    # evaluate
    mem_label = paddle.ones((mem_pred.shape[0], 1))
    non_mem_label = paddle.zeros((non_mem_pred.shape[0], 1))
    expected = paddle.concat([mem_label, non_mem_label], axis=0)
    eval_res = attack.evaluate(expected, result, metric_list=[Accuracy(), AUC(), Precision(), Recall()])

    print("""Evaluate result of ML-Leaks membership attack is: acc: {},
          auc: {}, precision: {}， recall: {}""".format(eval_res[0],
          eval_res[1], eval_res[2], eval_res[3]))

    print("Attack finish")


def train_model(model, train_data, test_data, epoch, learning_rate, batch_size):
    """
    train model
    """
    model.prepare(paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate),
                  paddle.nn.CrossEntropyLoss(),
                   [paddle.metric.Accuracy()])
    model.fit(train_data, test_data, epochs=epoch, verbose=1, batch_size=batch_size)


if __name__ == "__main__":
    arguments = parse_args()
    print("args: ", arguments)
    train_and_attack(arguments)


