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
temp script
"""
import sys
sys.path.append("..")
import numpy as np
import paddle
from attacks.gradient_method import FGSM, PGD
from attacks.cw import CW_L2
from models.whitebox import PaddleWhiteBoxModel
from defences.adversarial_transform import ClassificationAdversarialTransform

from classifier.definednet import transform_train, TowerNet
model_0 = TowerNet(3, 10, wide_scale=1)
model_1 = TowerNet(3, 10, wide_scale=2)

# set fgsm attack configuration
fgsm_attack_config = {"norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
paddle_model = PaddleWhiteBoxModel(
    [model_0, model_1],  # ensemble two models
    [1, 1.8],  # dictate weight
    paddle.nn.CrossEntropyLoss(),
    (-3, 3),
    channel_axis=3,
    nb_classes=10)

# "p" controls the probability of this enhance.
# for base model training, we set "p" == 0, so we skipped adv trans data augmentation.
# for adv trained model, we set "p" == 0.05, which means each batch
# will probably contain 5% adv trans augmented data.
enhance_config = {"p": 0.1, "norm_ord": np.inf, "epsilons": 0.0005, "epsilon_steps": 1, "steps": 1}
enhance_config2 = {"p": 0.1, "norm_ord": np.inf, "epsilons": 0.001, "epsilon_steps": 3, "steps": 3}
init_config3 = {"learning_rate": 0.01}
enhance_config3 = {"p": 0.05,
                   "attack_iterations": 15,
                   "c_search_steps": 6,
                   "verbose": False}

adversarial_trans = ClassificationAdversarialTransform(paddle_model,
                                                       [FGSM, PGD, CW_L2],
                                                       [None, None, init_config3],
                                                       [enhance_config, enhance_config2, enhance_config3])

cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
train_loader = paddle.io.DataLoader(cifar10_train, batch_size=16)

for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = paddle.unsqueeze(data[1], 1)
    x_data_augmented, y_data_augmented = adversarial_trans(x_data.numpy(), y_data.numpy())
    print(batch_id)
