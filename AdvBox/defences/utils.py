# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
utility tools for adversarial training.
"""
import sys
import os
sys.path.append("../..")

import paddle
import paddle.distributed as dist
import random
import numpy as np
import argparse
import logging


logger_initialized = []


def setup_logger(name="ppdet"):
    """
    Initialize logger and set its verbosity level to INFO.
    Args:
        name (str): the root module name of this logger
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    local_rank = dist.get_rank()
    if local_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger_initialized.append(name)
    return logger


def init_para_env():
    USE_GPU = paddle.get_device()
    if USE_GPU.startswith('gpu'):
        paddle.set_device("gpu")
    else:
        # raise ValueError('Distributed training only support GPU')
        paddle.set_device("cpu")

    env = os.environ
    distribute = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if distribute:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)
        paddle.seed(local_seed)
    dist.init_parallel_env()


def save_model(model, opt, save_dir, save_name, last_epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    paddle.save(model.state_dict(), save_path + ".pdparams")
    state_dict = opt.state_dict()
    state_dict['last_epoch'] = last_epoch
    paddle.save(state_dict, save_path + ".pdopt")


def resume_weights(model, model_path, optimizer=None, opt_path=None):
    # support Distill resume weights
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    param_state_dict = paddle.load(model_path)
    model_dict = model.state_dict()
    model_weight = {}

    for key in model_dict.keys():
        if key in param_state_dict.keys():
            model_weight[key] = param_state_dict[key]
        else:
            logger.info('Unmatched key: {}'.format(key))

    model.set_dict(model_weight)

    last_epoch = 0
    if optimizer is not None and opt_path is not None:
        opt_path = os.path.join(os.path.dirname(__file__), opt_path)
        optim_state_dict = paddle.load(opt_path)
        # to solve resume bug, will it be fixed in paddle 2.0
        for key in optimizer.state_dict().keys():
            if not key in optim_state_dict.keys():
                optim_state_dict[key] = optimizer.state_dict()[key]
        if 'last_epoch' in optim_state_dict:
            last_epoch = optim_state_dict.pop('last_epoch')
        optimizer.set_state_dict(optim_state_dict)

    return last_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=['resnet18',
                 'resnet34',
                 'resnet50',
                 'mobilenet'],
        default='resnet50'
    )
    parser.add_argument(
        "--dataset",
        choices=['cifar10', 'mini-imagenet'],
        default='mini-imagenet'
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--opt",
        choices=['rmsprop', 'momentum', 'adam'],
        default="adam"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005
    )
    parser.add_argument(
        "--scheduler",
        choices=['cosine', 'piecewise', 'reduce'],
        default=None
    )
    parser.add_argument(
        "--regularizer",
        choices=['l1', 'l2'],
        default=None
    )
    parser.add_argument(
        "--gamma",
        help="specify the step size of awp",
        type=float,
        default=0.005
    )
    parser.add_argument(
        "--attack_prob",
        help="ratio of AE when training",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--weights",
        help="load pretrained weights",
        default=None
    )
    parser.add_argument(
        "--opt_weights",
        help="load optimizer parameters when resuming training",
        default=None
    )
    parser.add_argument(
        "--save_path",
        default='advtrain_awp'
    )
    parser.add_argument(
        "--warmup",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1
    )
    args = parser.parse_args()
    return args


def load_model(model_name, num_classes=100, pretrained=False):
    if model_name == 'resnet18':
        model = paddle.vision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet34':
        model = paddle.vision.models.resnet34(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet50':
        model = paddle.vision.models.resnet50(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'mobilenet':
        model = paddle.vision.models.mobilenet_v1(pretrained=pretrained, num_classes=num_classes)
    else:
        print('Please choose a valid model')
        exit(-1)
    return model


def eval_robustness(model, eval_dataset, verbose=True, acc=1, **kwargs):
    from defences.pgd_perturb import pgd
    loader = paddle.io.DataLoader(eval_dataset, batch_size=1)
    ori_metric = paddle.metric.Accuracy(topk=[acc])
    adv_metric = paddle.metric.Accuracy(topk=[acc])

    model.eval()

    for i, data in enumerate(loader):
        img, label = data
        norm_img = paddle.squeeze(img)

        # # Calculate Top5 Accuracy
        logits = model(paddle.unsqueeze(norm_img, axis=0))
        correct = ori_metric.compute(logits, label)
        ori_metric.update(correct)

        # attack starts
        adv = pgd(model, paddle.squeeze(img), label, **kwargs)
        adv_logits = model(paddle.unsqueeze(adv, axis=0))
        adv_correct = adv_metric.compute(adv_logits, label)
        adv_metric.update(adv_correct)

    ori_acc = ori_metric.accumulate()
    adv_acc = adv_metric.accumulate()
    adv_success = (ori_acc - adv_acc) / ori_acc
    if verbose:
        print('Model Accuracy: {}'.format(ori_acc))
        print('Model Accuracy in presence of Adv: {}'.format(adv_acc))
        print('Attack Succeed Rate: {}'.format(adv_success))

    return ori_acc, adv_acc, adv_success

