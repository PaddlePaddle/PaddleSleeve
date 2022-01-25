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
Implememt of DLG model inversion attack
ref paper: https://arxiv.org/pdf/1906.08935.pdf
"""


import time
import logging
import numpy
import paddle

from paddle import Tensor

from typing import List

import paddle.nn.functional as F
from .inversion_attack import InversionAttack

class DLGInversionAttack(InversionAttack):
    """
    The implementation of DLG attack
    """

    """
    Params:
        learning_rate(float): The learning rate of attacking training
        attack_epoch(int): The iterations of attacking training
        window_size(int): When batch size greater than 1, 
            we update single data roundly for each window size iterations
        return_epoch(int): Return reconstructed data every 'return_epoch' epochs
    """
    params = ["learning_rate",
             "attack_epoch",
             "window_size",
             "return_epoch"] 

    def __init__(self, net, param_grad, data_shape, label_shape):
        """
        construct DLGInversionAttack

        Args:
            net(Layer): input target model which has not be updated with param_grad
                notice that the model must can be twice differentiable
            param_grad(List(Tensor)): model gradient computed by target data
            data_shape(Tensor): shape of target data
            labels_shape(Tensor): shape of target labels
        """
        self.net = net
        self.param_grad = param_grad
        self.data_shape = data_shape
        self.label_shape = label_shape

    def set_params(self, **kwargs):
        """
        Set parameters for attacks

        Args:
            kwargs(dict): Parameters of dictionary type
        """
        super().set_params(**kwargs)
        self.__check_params()

    def reconstruct(self, **kwargs):
        """
        reconstruct target data by DLG inversion attact

        Returns:
            (Tensor): multiple reconstructed data, which shape is (num, (data_shape, label_shape))
        """
        return self._dlg_attack()

    def _dlg_attack(self):
        """
        internal implememt dlg attack
        """
        # Generate dummy target data and labels. 
        dummy_x = numpy.random.normal(0, 1, size=self.data_shape).astype("float32")
        dummy_y = numpy.zeros(shape=self.label_shape).astype("float32")

        # try to reveal label
        label_idx = self._attack_label(self.param_grad[0])
        dummy_y[:, :, label_idx] = 1.0

        dummy_x = paddle.to_tensor(dummy_x)
        dummy_y = paddle.to_tensor(dummy_y)
        
        dummy_x.stop_gradient = False
        dummy_y.stop_gradient = False

        # the time of starting attack
        start = time.time()

        iter_count = 0
        updater_idx = 0

        ret = []

        for iteration in range(self.attack_epoch):

            dummy_pred = self.net(dummy_x)

            loss = F.mse_loss(dummy_pred, dummy_y, reduction='none')
            
            p_grad = paddle.grad(loss, self.net.parameters(), retain_graph=True, create_graph=True)

            loss_grad = paddle.to_tensor(numpy.array([0.0])).astype('float64')
            loss_grad.stop_gradient = False
            for i in range(len(self.param_grad)):
                loss_grad = loss_grad + (F.mse_loss(p_grad[i], self.param_grad[i], reduction='sum') / p_grad[i].numel())
            
            x_grad = paddle.grad(loss_grad, dummy_x, retain_graph=True, allow_unused=False)
            y_grad = paddle.grad(loss_grad, dummy_y, retain_graph=True, allow_unused=False)
            
            # when batch_size of data greater than 1, alternately update each data individually
            if iter_count == self.window_size:
                updater_idx = (updater_idx + 1) % self.data_shape[0]
                iter_count = 0
            
            dummy_x[updater_idx] = paddle.subtract(dummy_x[updater_idx], self.learning_rate * x_grad[0][updater_idx])
            dummy_y[updater_idx] = paddle.subtract(dummy_y[updater_idx], self.learning_rate * y_grad[0][updater_idx])

            iter_count = iter_count + 1
            
            # append results per 'return_epch' iterations
            if iteration % self.return_epoch == 0:
                ret.append((dummy_x.clone(), dummy_y.clone()))

            # reset tensor reference count for next iteration
            dummy_x = dummy_x.detach()
            dummy_y = dummy_y.detach()
            dummy_x.stop_gradient = False
            dummy_y.stop_gradient = False

        end = time.time()
        
        print("Attack cost time in seconds: {}".format(end - start))
        return ret

    def _attack_label(self, dw):
        """
        try to attack labels,
        ref to papper https://arxiv.org/pdf/2001.02610.pdf

        Args:
            dw(Tensor): gradient of weight
        """
        dw_mul = paddle.matmul(dw, dw, transpose_x=True)
        for i in range(dw_mul.shape[0]):
            dw_mul[i, i] = 0
            all_le_zero = all(x <= 0 for x in dw_mul[i])
            if all_le_zero:
                return i
        return 0

    def __check_params(self) -> None:
        """
        check params and set params default value
        """
        if not isinstance(self.learning_rate, float) or self.learning_rate < 0:
            raise ValueError("The parameter of learning rate must be a non-negative float value.")

        if not isinstance(self.attack_epoch, (int, numpy.int)) or self.attack_epoch < 0:
            raise ValueError("The parameter of attack epoch must be a non-negative integer value.")

        if not isinstance(self.window_size, (int, numpy.int)) or self.window_size < 0:
            raise ValueError("The parameter of window size must be a non-negative integer value.")

        if not isinstance(self.return_epoch, (int, numpy.int)) or self.return_epoch < 0:
            raise ValueError("The parameter of return epoch must be a positive integer value.")

        if not isinstance(self.param_grad, list) and not isinstance(self.param_grad[0], Tensor):
            raise ValueError("The input parameter param_grad must be a list of paddle Tensor value.")

        if not isinstance(self.net, paddle.nn.Layer):
            raise ValueError("The input parameter net must be a paddle Layer value.")
