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
from paddle.regularizer import L2Decay
print(paddle.__version__)
from defences.advtrain_base import adverarial_train_base
from defences.advtrain_trades import adverarial_train_trades
from defences.advtrain_awp import adversarial_train_awp
from attacks.gradient_method import FGSM
from attacks.logits_dispersion import LOGITS_DISPERSION

"""
According to the DL theory, the adversarial training is similar to adding a regularization
term on the training loss function. Thus, by controlling the adversarial enhance config to avoid
under-fitting in adversarial training process is important. Sometimes, in order to find a more
robust model in adversarial training, we have to adjust model structure (wider or deeper).
"""

#################################################################################################################
# CHANGE HERE: try different data augmentation methods and model type.
model_zoo = ("towernet", "preactresnet", "mobilenet", "resnet")
training_zoo = ("base", "advtraining", "advtraining_TRADES_FGSM", "advtraining_TRADES_LD", "advtraining_AWP_FGSM")
model_choice = input("choose {model_zoo}:".format(model_zoo=model_zoo))
training_choice = input("choose {training_zoo}:".format(training_zoo=training_zoo))
assert model_choice in model_zoo
assert training_choice in training_zoo

MODEL_PARA_NAME = training_choice + '_net_'
MODEL_OPT_PARA_NAME = training_choice + '_optimizer_'
if model_choice == 'towernet':
    # training process value
    EPOCH_NUM = 80
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 256
    advtrain_settings = {"epoch_num": EPOCH_NUM,
                         "advtrain_start_num": ADVTRAIN_START_NUM,
                         "batch_size": BATCH_SIZE,
                         "model_para_name": MODEL_PARA_NAME,
                         "model_opt_para_name": MODEL_OPT_PARA_NAME}

    from examples.classifier.definednet import transform_train, transform_eval, MEAN, STD, TowerNet
    # TowerNet
    if training_choice == training_zoo[0]:
        attack_method = None
        adverarial_train = adverarial_train_base
        init_config = None
        # "p" controls the probability of this enhance.
        # for base model training, we set "p" == 0, so we skipped adv trans data augmentation.
        enhance_config = {"p": 0}
        model = TowerNet(3, 10, wide_scale=1)
        opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    elif training_choice == training_zoo[1]:
        attack_method = FGSM
        adverarial_train = adverarial_train_base
        init_config = None
        # for adv trained model, we set "p" == 0.05, which means each batch
        # will probably contain 3% adv trans augmented data.
        enhance_config = {"p": 0.1}
        model = TowerNet(3, 10, wide_scale=1)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
    elif training_choice == training_zoo[2]:
        attack_method = FGSM
        adverarial_train = adverarial_train_trades
        init_config = None
        # 100% of each input batch will be convert into adv augmented data.
        enhance_config = {"p": 1}
        model = TowerNet(3, 10, wide_scale=1)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
    elif training_choice == training_zoo[3]:
        attack_method = LOGITS_DISPERSION
        adverarial_train = adverarial_train_trades
        init_config = {"norm": "Linf"}
        enhance_config = {"p": 1, "steps": 10, "dispersion_type": "softmax_kl", "verbose": False}
        model = TowerNet(3, 10, wide_scale=1)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
    else:
        exit(0)
    advtrain_settings["optimizer"] = opt

elif model_choice == 'mobilenet':
    # training process value
    EPOCH_NUM = 24
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 1024
    advtrain_settings = {"epoch_num": EPOCH_NUM,
                         "advtrain_start_num": ADVTRAIN_START_NUM,
                         "batch_size": BATCH_SIZE,
                         "model_para_name": MODEL_PARA_NAME,
                         "model_opt_para_name": MODEL_OPT_PARA_NAME}

    from examples.classifier.mobilenet_v3 import transform_train, transform_eval, MEAN, STD
    from examples.classifier.mobilenet_v3 import MobileNetV3_large_x1_0, MobileNetV3_large_x1_25
    path = '../classifier/pretrained_weights/MobileNetV3_large_x1_0_imagenet1k_pretrained.pdparams'
    model_state_dict = paddle.load(path)
    # MobileNet V3
    if training_choice == "base":
        adverarial_train = adverarial_train_base
        init_config = None
        enhance_config = {"p": 0}
        model = MobileNetV3_large_x1_0(class_dim=10)
        model.set_state_dict(model_state_dict)
    elif training_choice == "advtraining":
        adverarial_train = adverarial_train_base
        init_config = None
        enhance_config = {"p": 0.05}
        # adv trained model
        with paddle.utils.unique_name.guard():
            model = MobileNetV3_large_x1_0(class_dim=10)
    else:
        exit(0)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.004, T_max=5, verbose=True)
    opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    advtrain_settings["optimizer"] = opt

elif model_choice == 'resnet':
    # training process value
    EPOCH_NUM = 5
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 256
    advtrain_settings = {"epoch_num": EPOCH_NUM,
                         "advtrain_start_num": ADVTRAIN_START_NUM,
                         "batch_size": BATCH_SIZE,
                         "model_para_name": MODEL_PARA_NAME,
                         "model_opt_para_name": MODEL_OPT_PARA_NAME}

    from examples.classifier.resnet_vd import transform_train, transform_eval, MEAN, STD, ResNet50_vd
    path = '../classifier/pretrained_weights/ResNet50_vd_ssld_pretrained.pdparams'
    model_state_dict = paddle.load(path)
    # ResNet V50
    if training_choice == "base":
        adverarial_train = adverarial_train_base
        init_config = None
        enhance_config = {"p": 0}
        model = ResNet50_vd(class_dim=10)
        model.set_state_dict(model_state_dict)
    elif training_choice == "advtraining":
        adverarial_train = adverarial_train_base
        init_config = None
        enhance_config = {"p": 0.03}
        # adv trained model
        with paddle.utils.unique_name.guard():
            model = ResNet50_vd(class_dim=10)
            model.set_state_dict(model_state_dict)
    else:
        exit(0)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.001, T_max=10, verbose=True)
    opt = paddle.optimizer.Momentum(learning_rate=scheduler, momentum=0.9,
                                    parameters=model.parameters(), weight_decay=L2Decay(0.0001))
    advtrain_settings["optimizer"] = opt

elif model_choice == 'preactresnet':
    # training process value
    EPOCH_NUM = 80
    ADVTRAIN_START_NUM = 0
    BATCH_SIZE = 256
    advtrain_settings = {"epoch_num": EPOCH_NUM,
                         "advtrain_start_num": ADVTRAIN_START_NUM,
                         "batch_size": BATCH_SIZE,
                         "model_para_name": MODEL_PARA_NAME,
                         "model_opt_para_name": MODEL_OPT_PARA_NAME}

    from examples.classifier.preactresnet import transform_train, transform_eval, MEAN, STD, preactresnet18
    if training_choice == training_zoo[0]:
        attack_method = None
        adverarial_train = adverarial_train_base
        init_config = None
        enhance_config = {"p": 0}
        model = preactresnet18(num_classes=10)
        opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    elif training_choice == training_zoo[1]:
        attack_method = FGSM
        adverarial_train = adverarial_train_base
        init_config = None
        enhance_config = {"p": 0.1}
        model = preactresnet18(num_classes=10)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
    elif training_choice == training_zoo[2]:
        attack_method = FGSM
        adverarial_train = adverarial_train_trades
        init_config = None
        enhance_config = {"p": 1}
        model = preactresnet18(num_classes=10)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
        advtrain_settings["TRADES_beta"] = 1
    elif training_choice == training_zoo[3]:
        attack_method = LOGITS_DISPERSION
        adverarial_train = adverarial_train_trades
        init_config = {"norm": "Linf"}
        enhance_config = {"p": 1, "steps": 10, "dispersion_type": "softmax_kl", "verbose": False}
        model = preactresnet18(num_classes=10)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
        advtrain_settings["TRADES_beta"] = 1
    elif training_choice == training_zoo[4]:
        attack_method = FGSM
        adverarial_train = adversarial_train_awp
        init_config = {"norm": "Linf"}
        enhance_config = {"p": 0.5, "steps": 10, "dispersion_type": "softmax_kl", "verbose": False}
        model = preactresnet18(num_classes=10)
        opt = paddle.optimizer.Adam(learning_rate=0.0005, parameters=model.parameters())
        # TODO: undecided where to put.
        # advtrain_settings["AWP_adversary"] = AdvWeightPerturb(MODEL, gamma=0.005)
    else:
        exit(0)
    advtrain_settings["optimizer"] = opt

else:
    exit(0)

#################################################################################################################
MODEL = model
# all model weights will be saved under MODEL_PATH
p = enhance_config['p']
MODEL_PATH = '../cifar10/' + str(model_choice) + '_' + str(training_choice) + '_tutorial_result/'
# dataset
MEAN = MEAN
STD = STD
cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform_eval)
#################################################################################################################
