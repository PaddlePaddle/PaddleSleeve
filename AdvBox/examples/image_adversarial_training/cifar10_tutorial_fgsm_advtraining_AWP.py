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
import copy
from collections import OrderedDict
from attacks.gradient_method import FGSM
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

def adverarial_train_AWP(model, cifar10_train, cifar10_test, save_path=None, **kwargs):
    """
    A demo for adversarial training based on data augmentation.
    Reference Implementation: https://arxiv.org/abs/2004.05884
    Adversarial Weight Perturbation Helps Robust Generalization.

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
    awp_adversary = ADVTRAIN_SETTINGS["AWP_adversary"]
    opt = kwargs["optimizer"]
    train_loader = paddle.io.DataLoader(cifar10_train, shuffle=True, batch_size=batch_size)
    valid_loader = paddle.io.DataLoader(cifar10_test, batch_size=batch_size)
    max_acc = 0

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.unsqueeze(data[1], 1)

            # adversarial training late start
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

            # step 1: awp perturb
            awp = awp_adversary.calc_awp(inputs_adv=x_data_augmented, targets=y_data_augmented)
            awp_adversary.perturb(awp)

            logits = model(x_data_augmented)
            loss = F.cross_entropy(logits, y_data_augmented)
            acc = paddle.metric.accuracy(logits, y_data_augmented)
            acc = acc.numpy()
            acc = round(acc[0], 3)
            if batch_id % 10 == 0:
                print("epoch:{}, batch_id:{}, loss:{}, acc:{}".format(epoch, batch_id, loss.numpy(), acc))
            loss.backward()
            opt.step()
            opt.clear_grad()

            # step 2: awp restore
            awp_adversary.restore(awp)

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


# Below is the core module for AWP.
EPS = 1e-20

def diff_in_weights(model_state_dict, proxy_state_dict):
    diff_dict = OrderedDict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.shape) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    state_dict = model.state_dict()
    with paddle.no_grad():
        for name, param in state_dict.items():
            if name in names_in_diff:
                state_dict[name] += coeff * diff[name]
    model.set_state_dict(state_dict)


class AdvWeightPerturb(object):
    def __init__(self, model, gamma, loss='at_loss'):
        """
        :param model: A Paddlepaddle model which supports state_dict() and set_state_dict()
        :param gamma: The relative size of weight perturbation
        :param loss: The loss should be the same as the training loss (at_loss | trades_loss).
        """
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.awp_optim = paddle.optimizer.SGD(learning_rate=0.001, parameters=self.model.parameters())
        self.gamma = gamma
        self.loss = loss

        self.orig_state_dict = None

    def calc_awp(self, inputs_adv, targets):
        self.orig_state_dict = copy.deepcopy(self.model.state_dict())
        self.model.train()

        if self.loss == 'at_loss':
            loss = -F.cross_entropy(self.model(inputs_adv), targets)
        elif self.loss == 'trades_loss':
            loss = None
        else:
            raise ValueError('Please use the valid loss, ( at_loss | trades_loss )')

        self.awp_optim.clear_grad()
        loss.backward()
        self.awp_optim.step()

        diff = diff_in_weights(self.orig_state_dict, self.model.state_dict())
        self.model.set_state_dict(self.orig_state_dict)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

# Above is the core module for AWP.


def main():
    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [MODEL],
        [1],
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
        loss=paddle.nn.CrossEntropyLoss(),
        bounds=(-3, 3),
        input_channel_axis=0,
        input_shape=(3, 32, 32),
        nb_classes=10)
    adversarial_trans = ClassificationAdversarialTransform(paddle_model, [FGSM], [None], [ENHANCE_CONFIG])
    ADVTRAIN_SETTINGS["adversarial_trans"] = adversarial_trans
    ADVTRAIN_SETTINGS["AWP_adversary"] = AdvWeightPerturb(MODEL, gamma=0.005)
    val_acc_history, val_loss_history = adverarial_train_AWP(MODEL, CIFAR10_TRAIN, CIFAR10_TEST,
                                                             save_path=MODEL_PATH, **ADVTRAIN_SETTINGS)


if __name__ == '__main__':
    main()
