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
Main settings for adversarial training tutorials.
Contains:
* PreactResnet adversarial training benchmark on Cifar10 & Mini-ImageNet.
* Towernet finetuning with PGD advtraining mode on Mini-ImageNet.
* The other experiments to be finished.
"""
import os

import paddle
from paddle.regularizer import L2Decay
print(paddle.__version__)
from defences.advtrain_natural import adversarial_train_natural
from defences.advtrain_trades import adverarial_train_trades
from attacks.gradient_method import FGSM, PGD
from attacks.logits_dispersion import LD

"""
According to the DL theory, the adversarial training is similar to adding a regularization
term on the training loss function. Thus, by controlling the adversarial enhance config to avoid
under-fitting in adversarial training process is important. Sometimes, in order to find a more
robust model in adversarial training, we have to adjust model structure (wider or deeper).
"""

#################################################################################################################
# CHANGE HERE: try different data augmentation methods and model type.
# TODO: use parse_args...
model_zoo = ("towernet", "preactresnet")
training_zoo = ("base", "advtraining_natural", "advtraining_TRADES")
dataset_zoo = ("cifar10", "mini-imagenet")
attack_zoo = ("FGSM", "LD", "PGD")
use_base_pretrain_zoo = ("yes", "no")

def assert_input(model_choice, training_choice, dataset_choice, attack_choice, use_base_pretrain):
    assert model_choice in model_zoo, 'Only support model in {model_zoo}.'.format(model_zoo=model_zoo)
    assert training_choice in training_zoo, 'Only support training method in {training_zoo}.'.format(training_zoo=training_zoo)
    assert dataset_choice in dataset_zoo, 'Only support dataset in {dataset_zoo}.'.format(dataset_zoo=dataset_zoo)
    assert attack_choice in attack_zoo, 'Only support attack method in {attack_zoo}'.format(attack_zoo=attack_zoo)
    assert use_base_pretrain in use_base_pretrain_zoo, 'use_base_pretrain only support in {use_base_pretrain_zoo}'.format(use_base_pretrain_zoo=use_base_pretrain_zoo)
    

def get_mean_and_std(dataset_choice):
    if dataset_choice == 'cifar10':
        MEAN = [0.491, 0.482, 0.447]
        STD = [0.247, 0.243, 0.262]
    elif dataset_choice == 'mini-imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    return MEAN, STD

# Model Setting
def get_model_setting(model_choice, dataset_choice):

    if model_choice == 'towernet':
        from examples.classifier.towernet import get_transform, TowerNet
        if dataset_choice == dataset_zoo[0]:
            model = TowerNet(3, 10, wide_scale=1)
        elif dataset_choice == dataset_zoo[1]:
            model = TowerNet(3, 100, wide_scale=1)

    elif model_choice == 'preactresnet':
        from examples.classifier.preactresnet import get_transform, preactresnet18
        if dataset_choice == dataset_zoo[0]:
            model = preactresnet18(num_classes=10)
        elif dataset_choice == dataset_zoo[1]:
            model = preactresnet18(num_classes=100)

    mean, std = get_mean_and_std(dataset_choice)
    transform_train = get_transform(mean, std, 'train')
    transform_eval = get_transform(mean, std, 'eval')

    return model, mean, std, transform_train, transform_eval


def get_save_path(model_choice, training_choice, dataset_choice, attack_choice, use_base_pretrain):
    if training_choice == 'base':
        save_path = "./tutorial_result/%s/%s/%s/" % (dataset_choice, model_choice, training_choice)
    else:
        if use_base_pretrain == 'yes':
            save_path = "./tutorial_result/%s/%s/%s_finetuned/%s/" % (dataset_choice, model_choice, training_choice, attack_choice)
        else:
            save_path = "./tutorial_result/%s/%s/%s/%s/" % (dataset_choice, model_choice, training_choice, attack_choice)
    return save_path

# Attack Setting
def get_attack_setting(attack_choice):
    if attack_choice == 'FGSM':
        attack_method = FGSM
        init_config = {"norm": "Linf", "epsilon_ball": 8/255, "epsilon_stepsize": 2/255}
        attack_config = {}
    elif attack_choice == 'LD':
        attack_method = LD
        init_config = {"norm": "Linf", "epsilon_ball": 8/255}
        attack_config = {"steps": 10, "dispersion_type": "softmax_kl", "verbose": False}
    elif attack_choice == 'PGD':
        attack_method = PGD
        init_config = {"norm": "Linf", "epsilon_ball": 8 / 255, "epsilon_stepsize": 2 / 255}
        attack_config = {}
    return attack_method, init_config, attack_config

def get_model_para_name(training_choice):
    return training_choice + '_net_'

def get_opt_para_name(training_choice):
    return training_choice + '_optimizer_'

# Training Setting
def get_train_method_setting(model, training_choice):

    # Training Process Value Setting
    MODEL_PARA_NAME = get_model_para_name(training_choice)
    MODEL_OPT_PARA_NAME = get_opt_para_name(training_choice)
    EPOCH_NUM = 80
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 256

    advtrain_settings = {
        "epoch_num": EPOCH_NUM,
        "advtrain_start_num": ADVTRAIN_START_NUM,
        "batch_size": BATCH_SIZE,
        "model_para_name": MODEL_PARA_NAME,
        "model_opt_para_name": MODEL_OPT_PARA_NAME
    }

    # Train method setting
    if training_choice == 'base':
        adverarial_train = adversarial_train_natural
        # "p" controls the probability of this enhance.
        # for base model training, we set "p" == 0, so we skipped adv trans data augmentation.
        enhance_config = {"p": 0}

    elif training_choice == 'advtraining_natural':
        adverarial_train = adversarial_train_natural
        enhance_config = {"p": 0.1}

    elif training_choice == 'advtraining_TRADES':
        adverarial_train = adverarial_train_trades
        enhance_config = {"p": 1}
        advtrain_settings["TRADES_beta"] = 1

    if training_choice == 'base':
        opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    else:
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
    advtrain_settings["optimizer"] = opt
    return adverarial_train, enhance_config, advtrain_settings

# Dataset Setting
def get_dataset(dataset_choice, mode, transform):
    assert dataset_choice in dataset_zoo, 'Only support dataset in {dataset_zoo}.'.format(dataset_zoo=dataset_zoo)
    if dataset_choice == dataset_zoo[0]:
        dataset = paddle.vision.datasets.Cifar10(mode=mode, transform=transform)
        class_num = 10
    elif dataset_choice == dataset_zoo[1]:
        from examples.dataset.mini_imagenet import MINIIMAGENET
        if mode == 'train':
            cached_path = '../dataset/mini-imagenet/re_split_mini-imagenet-cache-train.pkl'
        elif mode == 'test':
            cached_path = '../dataset/mini-imagenet/re_split_mini-imagenet-cache-test.pkl'
        else:
            print('mini-imagenet only support mode in ["train", "test"].')
            exit(0)
        label_path = '../dataset/mini-imagenet/re_split_mini-imagenet_labels.txt'
        dataset = MINIIMAGENET(dataset_path=cached_path, label_path=label_path, mode=mode, transform=transform)
        class_num = 100
    return dataset, class_num
