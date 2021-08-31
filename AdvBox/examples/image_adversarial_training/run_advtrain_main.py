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
paddle2 model adversarial training demo on CIFAR10 data
"""
import sys
sys.path.append("../..")
import paddle
from defences.adversarial_transform import ClassificationAdversarialTransform
from models.whitebox import PaddleWhiteBoxModel

from main_setting import train_set, test_set, adverarial_train, advtrain_settings
TRAIN_SET = train_set
TEST_SET = test_set
ADVTRAIN_SETTINGS = advtrain_settings
ADVSERARIAL_TRAIN = adverarial_train

from main_setting import init_config, enhance_config, attack_method
INIT_CONFIG = init_config
ENHANCE_CONFIG = enhance_config
ATTACK_METHOD = attack_method

from main_setting import MODEL, MODEL_PATH, MODEL_PARA_NAME, MODEL_OPT_PARA_NAME, MEAN, STD, CLASS_NUM
MODEL = MODEL
MODEL_PATH = MODEL_PATH
MODEL_PARA_NAME = MODEL_PARA_NAME
MODEL_OPT_PARA_NAME = MODEL_OPT_PARA_NAME
CLASS_NUM = CLASS_NUM


def main():
    """
    Main function for running adversarial training.
    Returns:
        None
    """
    test_loader = paddle.io.DataLoader(TEST_SET, batch_size=1)
    data = test_loader().next()

    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [MODEL],
        [1],
        (0, 1),
        mean=MEAN,
        std=STD,
        input_channel_axis=0,
        input_shape=tuple(data[0].shape[1:]),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=CLASS_NUM)

    adversarial_trans = ClassificationAdversarialTransform(paddle_model, [ATTACK_METHOD],
                                                           [INIT_CONFIG], [ENHANCE_CONFIG])
    ADVTRAIN_SETTINGS["adversarial_trans"] = adversarial_trans
    val_acc_history, val_loss_history = ADVSERARIAL_TRAIN(MODEL, TRAIN_SET, TEST_SET,
                                                          save_path=MODEL_PATH, **ADVTRAIN_SETTINGS)


if __name__ == '__main__':
    main()
