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
Implememt of GAN model inversion attack
ref paper: https://arxiv.org/pdf/1702.07464.pdf
"""


import time
import logging
import numpy as np
import paddle

from paddle import Tensor

from typing import List

import paddle.nn.functional as F
from .inversion_attack import InversionAttack

class GANInversionAttack(InversionAttack):
    """
    The implementation of GAN attack
    """

    """
    Params:
        learning_rate_real(float): The learning rate of training dist_net with real data
        learning_rate_fake(float): The learning rate of training dist_net with fake data
        learning_rate_gen(float): The learning rate of training gen_net with fake data
    """
    params = ["learning_rate_real",
             "learning_rate_fake",
             "learning_rate_gen"] 

    def __init__(self, gen_net, label_size, target_label, target_fake_label, fake_dataload):
        """
        construct GANInversionAttack

        Args:
            gen_net(Layer): Generator for GAN
            label_size(int): Size of data label
            target_label(int): Label for attacking
            target_fake_label: Fake label for train dist net with fake data
            fake_dataload(DataLoader): Dataloader for fake data
        """
        self.gen_net = gen_net
        self.fake_label_shape = fake_dataload().next()[1].shape
        self.label_size = label_size
        self.target_label = target_label
        self.target_fake_label = target_fake_label
        self.fake_dataload = fake_dataload
        self.real_acc_set = []
        self.fake_avg_cost_set = []
        self.real_avg_cost_set = []
        self.g_avg_cost_set = []
        self.pass_id = 0

    def set_params(self, **kwargs):
        """
        Set parameters for attacks

        Args:
            kwargs(dict): Parameters of dictionary type
        """
        super().set_params(**kwargs)
        self.__check_params()

    def fit(self, model, data=None, **kwargs):
        """
        Train generator and discriminator (the input model)

        Args:
            model(Layer): input discriminator, also is target model
            data([data, label]): optional input real data for train dicriminator,
                if is None, only fake data is used to train dicriminator
            kwargs(dict): set key of "epoch" to make logging message per epoch, otherwise per step,
                i.g., kwargs = {"epoch": epoch_i}

        Returns:
            (Layer): trained discriminator
        """
        return self._GAN_attack(model, data, **kwargs)

    def reconstruct(self, **kwargs):
        """
        reconstruct target data by GAN inversion attact

        Returns:
            (Tensor): reconstructed data
        """
        return self.gen_net(self.fake_dataload().next()[0])

    def _GAN_attack(self, disc_net, data, **kwargs):
        """
        internal implememt gan attack
        """
        if kwargs.__contains__("epoch") and kwargs["epoch"] > self.pass_id:
            # if kwargs has key of 'epoch', print loss every epoch
            print("Attacker epoch: %d, fake_avg_cost: %f, "
                  "real_avg_cost: %f, real acc: %f, "
                  "g_avg_cost: %f"
                % (kwargs["epoch"], np.mean(self.fake_avg_cost_set),
                float("nan") if data is None else np.mean(self.real_avg_cost_set),
                float("nan") if data is None else np.mean(self.real_acc_set),
                np.mean(self.g_avg_cost_set)))
            self.real_acc_set = []
            self.fake_avg_cost_set = []
            self.real_avg_cost_set = []
            self.g_avg_cost_set = []
            self.pass_id = kwargs["epoch"]
        
        if data is not None:
            real_image = data[0]
            real_batch_labels = data[1]
            real_batch_labels = real_batch_labels.reshape([real_batch_labels.shape[0]])
            real_labels = F.one_hot(real_batch_labels, self.label_size)

        fake_labels = np.zeros(
            shape=[self.fake_label_shape[0], self.label_size], dtype='float32')
        fake_labels[:, self.target_fake_label] = 1.0

        target_labels = np.zeros(shape=[self.fake_label_shape[0], self.label_size], dtype='float32')
        target_labels[:, self.target_label] = 1.0

        fake_labels = paddle.to_tensor(fake_labels)
        target_labels = paddle.to_tensor(target_labels)

        # train generator to generate Indistinguishable fake data
        g_cost = self._train_g(self.gen_net, disc_net, self.fake_dataload().next()[0], target_labels)
        self.g_avg_cost_set.append(g_cost.numpy()[0])

        # train discriminator to distinguish fake data
        fake_cost = self._train_d_fake(disc_net, self.gen_net, self.fake_dataload().next()[0], fake_labels)
        self.fake_avg_cost_set.append(fake_cost.numpy()[0])

        if data is not None:
            # train discriminator for normal data
            real_cost, real_acc = self._train_d_real(disc_net, real_image, real_labels)
            self.real_avg_cost_set.append(real_cost.numpy()[0])
            self.real_acc_set.append(real_acc.numpy()[0])

        if not kwargs.__contains__("epoch"):
            # if kwargs do not has key of 'epoch', print loss every step
            print("Attacker step: %d, fake_avg_cost: %f, "
                  "real_avg_cost: %f, real acc: %f, "
                  "g_avg_cost: %f"
                % (self.pass_id, np.mean(self.fake_avg_cost_set), 
                float("nan") if data is None else np.mean(self.real_avg_cost_set),
                float("nan") if data is None else np.mean(self.real_acc_set),
                np.mean(self.g_avg_cost_set)))
            self.real_acc_set = []
            self.fake_avg_cost_set = []
            self.real_avg_cost_set = []
            self.g_avg_cost_set = []
            self.pass_id += 1

        return disc_net

    def _train_d_real(self, disc_net, real_image, label):
        """
        Train discriminator with normal data
        """
        p_real = disc_net(real_image)
        real_avg_cost = F.binary_cross_entropy_with_logits(p_real, label)
        acc_fn = paddle.metric.Accuracy()
        real_acc = acc_fn.compute(p_real, label)
        optim_d_real = paddle.optimizer.Adam(learning_rate=self.learning_rate_real,
                                             parameters=disc_net.parameters())
        real_avg_cost.backward()
        optim_d_real.step()
        optim_d_real.clear_grad()
        return real_avg_cost, real_acc


    def _train_d_fake(self, disc_net, gen_net, z, fake_label):
        """
        Train discriminator to distinguish fake data
        """
        x_fake = gen_net(z)
        p_fake = disc_net(x_fake.detach())
        fake_avg_cost = F.binary_cross_entropy_with_logits(p_fake, fake_label)
        optim_d_fake = paddle.optimizer.Adam(learning_rate=self.learning_rate_fake,
                                             parameters=disc_net.parameters())
        fake_avg_cost.backward()
        optim_d_fake.step()
        optim_d_fake.clear_grad()
        return fake_avg_cost

    def _train_g(self, gen_net, disc_net, z, target_label):
        """
        Train generator to generate Indistinguishable fake data
        """
        fake = gen_net(z)
        p = disc_net(fake)
        g_avg_cost = F.binary_cross_entropy_with_logits(p, target_label)
        optim_g = paddle.optimizer.Adam(learning_rate=self.learning_rate_gen,
                                        parameters=gen_net.parameters())
        g_avg_cost.backward()
        optim_g.step()
        optim_g.clear_grad()
        return g_avg_cost

    def __check_params(self) -> None:
        """
        check params and set params default value
        """
        if not isinstance(self.learning_rate_real, float) or self.learning_rate_real < 0:
            raise ValueError("The parameter of learning rate real must be a non-negative float value.")
        if not isinstance(self.learning_rate_fake, float) or self.learning_rate_fake < 0:
            raise ValueError("The parameter of learning rate fake must be a non-negative float value.")
        if not isinstance(self.learning_rate_gen, float) or self.learning_rate_gen < 0:
            raise ValueError("The parameter of learning rate gen must be a non-negative float value.")
        if not isinstance(self.gen_net, paddle.nn.Layer):
            raise ValueError("The parameter of gen_net must be a paddle.nn.Layer value.")
        if not isinstance(self.fake_dataload, paddle.io.DataLoader):
            raise ValueError("The parameter of fake_dataload must be a paddle.io.DataLoader value.")
        if not isinstance(self.label_size, (int, np.int32)) or self.label_size <= 0:
            raise ValueError("The parameter of label_size must be a non-negative int value.")
        if not isinstance(self.target_fake_label, (int, np.int32)) or self.target_fake_label < 0:
            raise ValueError("The parameter of target_fake_label must be a non-negative int value.")
        if not isinstance(self.target_label, (int, np.int32)) or self.target_label < 0:
            raise ValueError("The parameter of target_label must be a non-negative int value.")

