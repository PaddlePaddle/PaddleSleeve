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
This module provides differentially private optimizers for PaddlePaddle2.0.
"""

import numpy as np
import paddle
from paddle.distribution import Normal
from paddle.fluid.framework import Variable
from paddle.optimizer.adadelta import Adadelta
from paddle.optimizer.adagrad import Adagrad
from paddle.optimizer.adam import Adam
from paddle.optimizer.adamax import Adamax
from paddle.optimizer.adamw import AdamW
from paddle.optimizer.lamb import Lamb
from paddle.optimizer.momentum import Momentum
from paddle.optimizer.rmsprop import RMSProp
from paddle.optimizer.sgd import SGD


__all__ = ["DPAdadelta", "DPAdagrad", "DPAdam", "DPAdamax", "DPAdamW", 
           "DPLamb", "DPMomentum", "DPRMSProp", "DPSGD"]


def make_optimizer_class(optimizer_cls):
    """
    Construct a DP optimizer class from an existing optimizer class.
    """

    class DPOptimizerClass(optimizer_cls):
        """
        Differentially private subclass of given optimizer class 'optimizer_cls'.

        """

        def __init__(self,
                     parameters=None,
                     grad_clip=None,
                     stddev=None,
                     *args,
                     **kwargs):
            """Initialize the DPOptimizerClass.
            Args:
                parameters (list, optional): List of ``Tensor`` to update to minimize ``loss``. \
                    This parameter is required in dygraph mode. \
                    The default value is None in static mode, at this time all parameters will be updated.
                grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
                    some derived class of ``GradientClipBase`` . There are three cliping strategies
                    ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
                    :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
                stddev: The stddev of the noise added to the sum of gradient.
            """

            super(DPOptimizerClass, self).__init__(
                parameters=parameters,
                grad_clip = grad_clip,
                *args, **kwargs)

            self._parameter_list = list(
                parameters) if parameters is not None else None
            self._grad_clip = grad_clip
            self.stddev = stddev

        
        def minimize(self,
                     loss,
                     startup_program=None,
                     parameters=None,
                     no_grad_set=None,
                     batch_size=1):
            """
            Add operations to minimize ``loss`` by updating ``parameters`` for DP optimizers.
            
            Args:
                loss (Tensor): A ``Tensor`` containing the value to minimize.
                startup_program (Program, optional): :ref:`api_fluid_Program` for
                    initializing parameters in ``parameters``. The default value
                    is None, at this time :ref:`api_fluid_default_startup_program` will be used.
                parameters (list, optional): List of ``Tensor`` or ``Tensor.name`` to update
                    to minimize ``loss``. The default value is None, at this time all parameters
                    will be updated.
                no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                    to be updated. The default value is None.

            Returns:
                tuple: tuple (optimize_ops, params_grads), A list of operators appended
                by minimize and a list of (param, grad) tensor pairs, param is
                ``Parameter``, grad is the gradient value corresponding to the parameter.
                In static graph mode, the returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to 
                indicate program pruning. If so, the program will be pruned by ``feed`` and 
                ``fetch_list`` before run, see details in ``Executor``.

            Examples:
                .. code-block:: python
     
                    import paddle
                    linear = paddle.nn.Linear(10, 10)
                    input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
                    out = linear(input)

                    clip = paddle.nn.ClipGradByNorm(clip_norm=10.0)

                    adam = paddle.optimizer.DPAdam(learning_rate=0.1,
                            parameters=linear.parameters(),
                            grad_clip=clip
                            stddev=1.0)
                    adam.minimize(loss)
                    adam.clear_grad()

            """
            assert isinstance(loss, Variable), "The loss should be an Tensor."

            parameter_list = parameters if parameters else self._parameter_list

            # create param list, grad list, grads_sum list
            params_grads = []
            grads = []
            grads_sum = []

            loss[0].backward(retain_graph=True)
            for param in parameter_list:
                if not param.trainable:
                    continue
                if param._grad_ivar() is not None or True:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
                    grads.append(grad_var)

                    grad_tmp = paddle.zeros(param.shape)
                    grads_sum.append(grad_tmp)

            # clear gradients
            self.clear_grad()

            # accumulate grad => grads_sum
            for i in range(batch_size):
                loss[i].backward(retain_graph=True) # compute gradient
                params_grads = self._grad_clip(params_grads) # clip gradient
                grads_sum = grads_sum + grads # accumulate gradient
                self.clear_grad() # clear gradient

            # generate noise list
            dist = Normal(loc=0, scale=self.stddev)
            for j in range(len(parameter_list)):
                noise = dist.sample(grads_sum[j].shape)
                grads[j] = grads[j] / batch_size + noise # average, add noise

            # apply gradient to update parameter
            optimize_ops = self._apply_optimize(
                loss, startup_program=startup_program, params_grads=params_grads) # descent

            return optimize_ops, params_grads

    return DPOptimizerClass


DPAdadelta = make_optimizer_class(Adadelta)
DPAdagrad = make_optimizer_class(Adagrad)
DPAdam = make_optimizer_class(Adam)
DPAdamax = make_optimizer_class(Adamax)
DPAdamW = make_optimizer_class(AdamW)
DPLamb = make_optimizer_class(Lamb)
DPMomentum = make_optimizer_class(Momentum)
DPRMSProp = make_optimizer_class(RMSProp)
DPSGD = make_optimizer_class(SGD)
