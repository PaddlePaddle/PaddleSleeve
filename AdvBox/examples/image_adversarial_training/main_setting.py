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
Main settings for adversarial training tutorial.
"""
import paddle
import numpy as np
from paddle.regularizer import L2Decay
print(paddle.__version__)


"""
According to the DL theory, the adversarial training is similar to adding a regularization
term on the training loss function. Thus, by controlling the adversarial enhance config to avoid
under-fitting in adversarial training process is important. Sometimes, in order to find a more
robust model in adversarial training, we have to adjust model structure (wider or deeper).
"""

#################################################################################################################
# CHANGE HERE: try different data augmentation methods and model type.
model_choice = input("choose towernet/mobilenet/resnet:")
training_choice = input("choose base/advtraining/advtraining_TRADES:")
assert model_choice in ("towernet", "mobilenet", "resnet")
assert training_choice in ("base", "advtraining", "advtraining_TRADES")
# inputs and labels are not required for dynamic graph.
if model_choice == 'towernet':
    from examples.classifier.definednet import transform_train, transform_eval, TowerNet
    # TowerNet
    if training_choice == "base":
        # "p" controls the probability of this enhance.
        # for base model training, we set "p" == 0, so we skipped adv trans data augmentation.
        enhance_config = {"p": 0, "norm_ord": np.inf, "epsilons": 0.005, "epsilon_steps": 1, "steps": 1}
        model = TowerNet(3, 10, wide_scale=1)
        opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    elif training_choice == "advtraining":
        # for adv trained model, we set "p" == 0.05, which means each batch
        # will probably contain 3% adv trans augmented data.
        enhance_config = {"p": 0.03, "norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
        model = TowerNet(3, 10, wide_scale=1)
        # experiment wide_scale=2 ^_^...
        # model = TowerNet(3, 10, wide_scale=2)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
    elif training_choice == "advtraining_TRADES":
        # 100% of each input batch will be convert into adv augmented data.
        enhance_config = {"p": 1, "norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
        # enhance_config = {"p": 1, "epsilon": 0.031, "perturb_steps": 10, "verbose": True}
        model = TowerNet(3, 10, wide_scale=1)
        # experiment wide_scale=2 ^_^...
        # model = TowerNet(3, 10, wide_scale=2)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
    else:
        exit(0)
    # training process value
    EPOCH_NUM = 60
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 256

elif model_choice == 'mobilenet':
    from examples.classifier.mobilenet_v3 import transform_train, transform_eval
    from examples.classifier.mobilenet_v3 import MobileNetV3_large_x1_0, MobileNetV3_large_x1_25
    path = '../classifier/pretrained_weights/MobileNetV3_large_x1_0_imagenet1k_pretrained.pdparams'
    model_state_dict = paddle.load(path)
    # MobileNet V3
    if training_choice == "base":
        enhance_config = {"p": 0, "norm_ord": np.inf, "epsilons": 0.005, "epsilon_steps": 1, "steps": 1}
        model = MobileNetV3_large_x1_0(class_dim=10)
        model.set_state_dict(model_state_dict)
    elif training_choice == "advtraining":
        enhance_config = {"p": 0.05, "norm_ord": np.inf, "epsilons": 0.0005, "epsilon_steps": 1, "steps": 1}
        # adv trained model
        with paddle.utils.unique_name.guard():
            model = MobileNetV3_large_x1_0(class_dim=10)
            # experiment MobileNetV3_large_x1_25 ^_^...
            # model = MobileNetV3_large_x1_25(class_dim=10)
            # model.set_state_dict(model_state_dict)
    else:
        exit(0)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.004, T_max=5, verbose=True)
    opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    # training process value
    EPOCH_NUM = 24
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 1024

elif model_choice == 'resnet':
    from examples.classifier.resnet_vd import transform_train, transform_eval, ResNet50_vd
    path = '../classifier/pretrained_weights/ResNet50_vd_ssld_pretrained.pdparams'
    model_state_dict = paddle.load(path)
    # ResNet V50
    if training_choice == "base":
        enhance_config = {"p": 0, "norm_ord": np.inf, "epsilons": 0.005, "epsilon_steps": 1, "steps": 1}
        model = ResNet50_vd(class_dim=10)
        model.set_state_dict(model_state_dict)
    elif training_choice == "advtraining":
        enhance_config = {"p": 0.03, "norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
        # adv trained model
        with paddle.utils.unique_name.guard():
            model = ResNet50_vd(class_dim=10)
            model.set_state_dict(model_state_dict)
    else:
        exit(0)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.001, T_max=10, verbose=True)
    opt = paddle.optimizer.Momentum(learning_rate=scheduler, momentum=0.9,
                                    parameters=model.parameters(), weight_decay=L2Decay(0.0001))
    # training process value
    EPOCH_NUM = 5
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 256
else:
    exit(0)

#################################################################################################################
MODEL_PARA_NAME = training_choice + '_net_'
MODEL_OPT_PARA_NAME = training_choice + '_optimizer_'
MODEL = model
# all model weights will be saved under MODEL_PATH
p = enhance_config['p']
epsilons = enhance_config['epsilons']
MODEL_PATH = '../cifar10/' + str(model_choice) + '_' + str(p) + '_' + str(epsilons) + '_tutorial_result/'
# adversarial training settings
advtrain_settings = {"epoch_num": EPOCH_NUM, "advtrain_start_num": ADVTRAIN_START_NUM, "batch_size": BATCH_SIZE, "optimizer": opt}
# dataset
cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform_eval)
#################################################################################################################
# FGSM attack evaluation settings
fgsm_attack_config = {"norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
TOTAL_TEST_NUM = 500
#################################################################################################################
