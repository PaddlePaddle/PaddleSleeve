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
import paddle.nn.functional as F
import numpy as np
from attacks.logits_dispersion import LOGITS_DISPERSION
from defences.adversarial_transform import ClassificationAdversarialTransform
from models.whitebox import PaddleWhiteBoxModel

from main_setting import cifar10_train, cifar10_test, advtrain_settings, enhance_config
CIFAR10_TRAIN = cifar10_train
CIFAR10_TEST = cifar10_test
ADVTRAIN_SETTINGS = advtrain_settings
ENHANCE_CONFIG = enhance_config

from main_setting import MODEL, MODEL_PATH, MODEL_PARA_NAME, MODEL_OPT_PARA_NAME
MODEL = MODEL
MODEL_PATH = MODEL_PATH
MODEL_PARA_NAME = MODEL_PARA_NAME
MODEL_OPT_PARA_NAME = MODEL_OPT_PARA_NAME

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def adverarial_train_TRADES(model, cifar10_train, cifar10_test, save_path=None, **kwargs):
    """
    A demo for adversarial training based on data augmentation.
    Reference Implementation: https://arxiv.org/abs/1901.08573
    Theoretically Principled Trade-off between Robustness and Accuracy.

    Args:
        model: paddle model.
        cifar10_train: paddle dataloader.
        cifar10_test: paddle dataloader.
        save_path: str. path for saving model.
        **kwargs: Other named arguments.
    Returns:
        training log
    """
    assert save_path is not None
    print('start training ... ')
    val_acc_history = []
    val_loss_history = []
    epoch_num = kwargs["epoch_num"]
    advtrain_start_num = kwargs["advtrain_start_num"]
    batch_size = kwargs["batch_size"]
    adversarial_trans = kwargs["adversarial_trans"]
    opt = kwargs["optimizer"]
    beta = kwargs["TRADES_beta"]
    kldiv_criterion = paddle.nn.KLDivLoss(reduction='batchmean')
    logsoftmax = paddle.nn.LogSoftmax()
    softmax = paddle.nn.Softmax()
    train_loader = paddle.io.DataLoader(cifar10_train, shuffle=True, batch_size=batch_size)
    valid_loader = paddle.io.DataLoader(cifar10_test, batch_size=batch_size)
    max_acc = 0
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.unsqueeze(data[1], 1)
            # adversarial training late start
            # TODO: use kl between x_adv and x to compute x_data_augmented. attack method should add.w
            if epoch >= advtrain_start_num and adversarial_trans is not None:
                x_data_augmented, y_data_augmented = adversarial_trans(x_data.numpy(), y_data.numpy())
            else:
                x_data_augmented, y_data_augmented = x_data, y_data
            # turn model into training mode
            model.train()
            # make sure gradient flow for model parameter
            for param in model.parameters():
                param.stop_gradient = False

            # numpy to paddle.Tensor
            x_data_augmented = paddle.to_tensor(x_data_augmented, dtype='float32', place=USE_GPU)
            y_data_augmented = paddle.to_tensor(y_data_augmented, dtype='int64', place=USE_GPU)
            y_data_augmented = paddle.unsqueeze(y_data_augmented, 1)

            logits = model(x_data)
            loss_ce = F.cross_entropy(logits, y_data_augmented)
            # guaranteed 100% of batch samples have been adv augmented in main_setting
            # regardless of attack success rate.
            logits_advs = model(x_data_augmented)
            loss_logits_kl = kldiv_criterion(logsoftmax(logits_advs), softmax(logits))
            loss = loss_ce + beta * loss_logits_kl

            acc = paddle.metric.accuracy(logits, y_data_augmented)
            acc_adv = paddle.metric.accuracy(logits_advs, y_data_augmented)
            acc, acc_adv = acc.numpy(), acc_adv.numpy()
            acc, acc_adv = round(acc[0], 3), round(acc_adv[0], 3)
            if batch_id % 10 == 0:
                print("epoch:{}, batch_id:{}, loss:{}, acc:{}, acc_adv:{}"
                      .format(epoch, batch_id, loss.numpy(), acc, acc_adv))
            loss.backward()
            opt.step()
            opt.clear_grad()
        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        with paddle.no_grad():
            for batch_id, data in enumerate(valid_loader()):
                x_data = data[0]
                y_data = paddle.unsqueeze(data[1], 1)
                logits = model(x_data)
                loss = F.cross_entropy(logits, y_data)
                acc = paddle.metric.accuracy(logits, y_data)
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        avg_acc = round(avg_acc, 6)
        avg_loss = round(avg_loss, 6)
        if avg_acc > max_acc:
            max_acc = avg_acc
            paddle.save(model.state_dict(), save_path + MODEL_PARA_NAME + str(max_acc) + '.pdparams')
            paddle.save(opt.state_dict(), save_path + MODEL_OPT_PARA_NAME + str(max_acc) + '.pdopt')
            print("best saved at: ", save_path)
        else:
            pass
        print("[validation] accuracy/loss:{}/{}, max_acc:{}".format(avg_acc, avg_loss, max_acc))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()

    return val_acc_history, val_loss_history


def main():
    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [MODEL],
        [1],
        paddle.nn.CrossEntropyLoss(),
        (-3, 3),
        channel_axis=3,
        nb_classes=10)
    adversarial_trans = ClassificationAdversarialTransform(paddle_model, [LOGITS_DISPERSION], [None], [ENHANCE_CONFIG])
    ADVTRAIN_SETTINGS["adversarial_trans"] = adversarial_trans
    ADVTRAIN_SETTINGS["TRADES_beta"] = 1
    val_acc_history, val_loss_history = adverarial_train_TRADES(MODEL, CIFAR10_TRAIN, CIFAR10_TEST,
                                                                save_path=MODEL_PATH, **ADVTRAIN_SETTINGS)


if __name__ == '__main__':
    main()
