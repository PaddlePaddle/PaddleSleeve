# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
This module provides DGC (deep gradient compress) Momentum optimizers for PaddlePaddle2.0.
"""

import numpy as np
import paddle

from paddle.fluid.framework import Variable

from paddle.optimizer.momentum import Momentum

__all__ = ["DGCMomentum"]


class DGCMomentum(Momentum):
    """
    DGC (Deep Gradient Compression) Momentum Optimizer. Original paper is https://arxiv.org/abs/1712.01887
    DGC only uses important gradients (gradients larger than a threshold) to update parameters.
    """

    def __init__(self,
                 learning_rate=0.001,
                 momentum=0.9,
                 sparsity=0.80,
                 parameters=None,
                 use_nesterov=False,
                 weight_decay=None,
                 rescale_grad=1.0,
                 grad_clip=None,
                 name=None):
        r"""
        Simple DGCMomentum optimizer with velocity state
        This optimizer has a flag for Nestrov DGCMomentum.

        Parameters:

        learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        momentum (float): Momentum factor. The default value is 0.9.
        sparsity (float): DGC sparsity, will filter sparsity gradient each step. The default value is 0.8.
        parameters (list|tuple, optional): List|Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        rescale_grad (float, optional): Multiply the gradient with `rescale_grad` before updating. \
            Often choose to be ``1.0/batch_size``.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .
            """

        super(DGCMomentum, self).__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            parameters=parameters,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            grad_clip = grad_clip,
            rescale_grad=rescale_grad,
            name=None)
        self.time_step = 0
        self.sparsity = sparsity

    def backward(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        calc backward op for parameters, override base class method
        """
        parameter_list = parameters if parameters \
                else self._parameter_list
        params_grads = super(DGCMomentum, self).backward(
                            loss,
                            startup_program,
                            parameter_list,
                            no_grad_set,
                            callbacks)
        params_grads = self._dgc(params_grads, self.time_step, self._momentum)
        self.time_step += 1
        return params_grads

    def _dgc(self, params_grads, time_step, m):
        self._accumulate_g = [g[1] for g in params_grads]
        if time_step == 0:
            self._accumulate_u = [0 for i in range(len(params_grads))]
            self._accumulate_v = [0 for i in range(len(params_grads))]
        if not self._use_nesterov:
            self._accumulate_v = [(v + g) for v, g in zip(self._accumulate_v, self._accumulate_g)]
            self._accumulate_u = [(u + v) for u, v in zip(self._accumulate_u, self._accumulate_v)]
            self._accumulate_v = [m * v for v in self._accumulate_v]
        else:
            self._accumulate_v = [m * (v + g) for v, g in zip(self._accumulate_v, self._accumulate_g)]
            self._accumulate_u = \
                [(u + v + g) for u, v, g in zip(self._accumulate_u, self._accumulate_v, self._accumulate_g)]
            
        for j in range(len(self._accumulate_u)):
            u = self._accumulate_u[j]
            num_u = u.numel()

            u_ = u.flatten().abs()
            thr = paddle.topk(u_, num_u - int(self.sparsity * num_u))[0][-1]
            thr_tensor = paddle.full(u.shape, thr, dtype="float32")
            mask = paddle.greater_than(self._accumulate_u[j].abs(), thr_tensor)
            
            params_grads[j] = (params_grads[j][0], u * mask)
            self._accumulate_u[j] = self._accumulate_u[j] * paddle.logical_not(mask)

        return params_grads
