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
Demonstrate Adversarial Weight Perturbation on mini-imagenet dataset
Reference Implementation: https://arxiv.org/abs/2004.05884
"""

import sys

sys.path.append("../..")

import paddle
import numpy as np
import paddle.vision.transforms as T
import paddle.distributed as dist
from defences.utils import *
from defences.pgd_perturb import PGDTransform
from defences.advtrain_awp import adversarial_train_awp
from examples.dataset.mini_imagenet import MINIIMAGENET

# Mini-Imagenet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    # Load dataset
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD, data_format='CHW')
    ])
    train_dataset_path = os.path.join(os.path.realpath(__file__ + "/../.."), 'dataset/mini-imagenet/re_split_mini-imagenet-cache-train.pkl')
    val_dataset_path = os.path.join(os.path.realpath(__file__ + "/../.."), 'dataset/mini-imagenet/re_split_mini-imagenet-cache-test.pkl')
    label_path = os.path.join(os.path.realpath(__file__ + "/../.."), 'dataset/mini-imagenet/re_split_mini-imagenet_labels.txt')

    train_dataset = MINIIMAGENET(dataset_path=train_dataset_path,
                                 label_path=label_path,
                                 mode='train',
                                 transform=transform)
    test_dataset = MINIIMAGENET(dataset_path=val_dataset_path,
                                label_path=label_path,
                                mode='val',
                                transform=transform)

    # Initialize parallel environment
    init_para_env()

    # Load model
    m = paddle.vision.models.resnet50(pretrained=False, num_classes=100)

    m = paddle.DataParallel(m)
    m.train()

    # Training config
    batch_size = 256
    attack_config = {'num_classes': 100, 'model_mean': MEAN, 'model_std': STD}
    advtrans = PGDTransform(m, attack_config, p=0.5)
    num_per_epoch = len(train_dataset) // (dist.get_world_size() * batch_size)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.001,
                                                 warmup_steps=int(num_per_epoch),
                                                 start_lr=1e-5,
                                                 end_lr=0.001)

    opt = paddle.optimizer.RMSProp(learning_rate=scheduler, parameters=m.parameters())

    kwargs = {'epoch_num': 60,
              'advtrain_start_num': 20,
              'batch_size': batch_size,
              'gamma': 0.02,
              'adversarial_trans': advtrans,
              'optimizer': opt,
              'lr': scheduler,
              'weights': None,
              'opt_weights': None}

    save_path = os.path.join(os.path.dirname(__file__), "output/mini_imagenet_demo_advtrain_awp")
    adversarial_train_awp(model=m,
                          train_set=train_dataset,
                          test_set=test_dataset,
                          save_path=save_path,
                          **kwargs)


if __name__ == '__main__':
    main()
