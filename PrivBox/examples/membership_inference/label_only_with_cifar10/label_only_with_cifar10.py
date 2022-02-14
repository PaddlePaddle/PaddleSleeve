#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
This module provides an example of Label-only membership inference attack.
"""

from __future__ import print_function

import sys
import os
sys.path.append('../../../')

import argparse
import numpy
import time

import paddle
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F

from inference.membership_inference import LabelOnlyMembershipInferenceAttack
from inference.membership_inference.label_only_ml_inf import check_correct, augmentation_attack_set
from metrics import MSE, AUC, Accuracy, Precision, Recall

def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("Label-Only")
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
    parser.add_argument("--shadow_dataset",
                        type=str, default="cifar10",
                        help="shadow dataset")
    parser.add_argument("--target_dataset",
                        type=str, default="cifar10",
                        help="target dataset")
    parser.add_argument("--shadow_model", type=str, choices=['resnet18', 'resnet34'],
                        default='resnet18',
                        help="using what shadow model resnet18 or resnet34")
    parser.add_argument("--attack_type", 
                        type=str, default='r',
                        help="Type of attack to perform, r is rotation")
    parser.add_argument('--r', type=int, default=6, help='r param in rotation attack if used')
    #parser.add_argument('--d', type=int, default=1, help='d param in translation attack if used')
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
    return sets 


def train_and_attack(args):
    """
    The training procedure that starts from training target model,
    then launchs Label-only membership inference attack

    Args:
        args(ArgumentParser): the execution parameters.
    """
    
    transform = Compose([paddle.vision.Resize((32, 32)),
                    Normalize(mean=[127.5], std=[127.5], data_format='CHW'),
                    paddle.vision.transforms.Transpose()])
    used_dataset = (args.shadow_dataset, args.target_dataset)

    # load choose datasets
    cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
    cifar10_train_data = split_dataset(cifar10_train, batch_size=args.batch_size)
    
    cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform)
    cifar10_test_data = split_dataset(cifar10_test, batch_size=args.batch_size)
    
    # get shadow and target data
    
    shadow_train_data = cifar10_train_data[1]
    shadow_test_data = cifar10_test_data[1]
    shadow_num_classes = 10

    target_train_data = cifar10_train_data[0]
    target_test_data = cifar10_test_data[0]
    target_num_classes = 10
    """
    shadow_train_data, _ = paddle.io.random_split(cifar10_train_data[1], [1000, len(cifar10_train_data[1])-1000]) 
    shadow_test_data, _ = paddle.io.random_split(cifar10_test_data[1], [1000, len(cifar10_test_data[1])-1000]) 
    shadow_num_classes = 10

    target_train_data, _ = paddle.io.random_split(cifar10_train_data[0], [1000, len(cifar10_train_data[0])-1000])
    target_test_data, _ = paddle.io.random_split(cifar10_test_data[0], [1000, len(cifar10_test_data[0])-1000]) 
    target_num_classes = 10
    """
    # define target model
    target_model = ResNet18(num_classes=target_num_classes)
    target_model = paddle.Model(target_model)
    print("Begin training target model")

    #train target model
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
    attack = LabelOnlyMembershipInferenceAttack(shadow_model, 
                                                shadow_dataset=[shadow_train_data, shadow_test_data])
    aug_kwarg = args.r
    attack_params = {"batch_size": args.batch_size, "shadow_epoch": args.shadow_epoch,
                     "classifier_epoch": args.classifier_epoch,
                     "shadow_lr":args.shadow_lr, "classifier_lr": args.classifier_lr,
                     "attack_type": args.attack_type, "aug_kwarg":aug_kwarg}

    # set params
    attack.set_params(**attack_params)

    print("Infer target dataset")
    #target data augment and predict
    target_pred = augmentation_attack_set(target_model, 
                                          target_train_data, 
                                          target_test_data, 
                                          args.batch_size, 
                                          args.attack_type, 
                                          aug_kwarg)
    mem_pred = target_pred[0][:len(target_train_data)]
    non_mem_pred = target_pred[0][len(target_train_data):]
    data = paddle.concat([mem_pred, non_mem_pred])

    result = attack.infer(data)
    
    # evaluate
    mem_label = paddle.ones((mem_pred.shape[0], 1))
    non_mem_label = paddle.zeros((non_mem_pred.shape[0], 1))
    expected = paddle.concat([mem_label, non_mem_label], axis=0)
    eval_res = attack.evaluate(expected, result, metric_list=[Accuracy(), AUC(), Precision(), Recall()])
    
    print("""Evaluate result of Label-only membership attack is: acc: {},
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
