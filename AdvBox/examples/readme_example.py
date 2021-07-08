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
import paddle
import numpy as np
from adversary import Adversary
from attacks.gradient_method import FGSM
from attacks.cw import CW_L2
from attacks.logits_dispersion import LOGITS_DISPERSION
from models.whitebox import PaddleWhiteBoxModel

from classifier.definednet import transform_eval, TowerNet
model_0 = TowerNet(3, 10, wide_scale=1)
model_1 = TowerNet(3, 10, wide_scale=2)

# set fgsm attack configuration
paddle_model = PaddleWhiteBoxModel(
    [model_0, model_1],   # ensemble two models
    [1, 1.8],             # dictate weight
    paddle.nn.CrossEntropyLoss(),
    (-3, 3),
    channel_axis=3,
    nb_classes=10)

# FGSM attack, init attack with the ensembled model
# attack = FGSM(paddle_model)
# attack = CW_L2(paddle_model, learning_rate=0.01)
attack = LOGITS_DISPERSION(paddle_model, dispersion_type='softmax_kl_l_inf')

cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform_eval)
test_loader = paddle.io.DataLoader(cifar10_test, batch_size=1)

data = test_loader().next()
img = data[0][0]
label = data[1]

# init adversary status
adversary = Adversary(img.numpy(), int(label))
# target = np.random.randint(paddle_model.num_classes())
# while label == target:
#     target = np.random.randint(paddle_model.num_classes())
# print(label, target)
# adversary.set_status(is_targeted_attack=True, target_label=target)

# launch attack
# adversary = attack(adversary, norm_ord=np.inf, epsilons=0.003, epsilon_steps=1, steps=1)
# adversary = attack(adversary, attack_iterations=100, verbose=True)
adversary = attack(adversary, epsilon=0.031, perturb_steps=10, verbose=True)

if adversary.is_successful():
    original_img = adversary.original
    adversarial_img = adversary.adversarial_example
    print("Attack succeeded.")
else:
    print("Attack failed.")
