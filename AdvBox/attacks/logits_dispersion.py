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
This module provides the attack method by model logits dispersion.
"""
from __future__ import division
from __future__ import print_function

from builtins import range
import math
import numpy as np
import paddle
from .base import Attack


__all__ = ['LOGITS_DISPERSION', 'LD']


class LOGITS_DISPERSION(Attack):
    """
    Uses certain optimizer or gradient to maximize distance between x's logits and x_adv's logits.
    It's usually used in adversarial training.
    Related Paper:
    1. Theoretically Principled Trade-off between Robustness and Accuracy.
    2. Adversarial Logit Pairing.
    3. Virtual Adversarial Training: a regularization method for supervised and semi-supervised learning.
    4. Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks.

    Including:
    * softmax_kl, softmax of logits KL dispersion
    * logits_norm, logits of norm distance dispersion, for instantce ||logit - logits_adv||22
    * difference_logits_ratio, Difference of Logits Ratio Loss. DLR(x, y) = − (zy − max(i!=y)zi) / (zπ1 − zπ3)
    Args:
        model: PaddleWhiteBoxModel.
        learning_rate: float. for adam optimizer.
    """
    def __init__(self, model, norm='Linf', epsilon_ball=8/255, epsilon_stepsize=2/255):
        super(LOGITS_DISPERSION, self).__init__(model, norm=norm,
                                                epsilon_ball=epsilon_ball,
                                                epsilon_stepsize=epsilon_stepsize)
        self.model = model
        self.safe_num = 0.999999
        # (float, float), It is used to map input float into (0, 1) for arctanh
        self.box_constrains_lower_bound = self.model.bounds[0]
        self.box_constrains_upper_bound = self.model.bounds[1]
        assert self.box_constrains_lower_bound < self.box_constrains_upper_bound

        self.support_type = ('softmax_kl', 'logits_norm', 'difference_logits_ratio')
        self.kldiv_criterion = paddle.nn.KLDivLoss(reduction='batchmean')
        self.logsoftmax = paddle.nn.LogSoftmax()
        self.softmax = paddle.nn.Softmax()

    def _apply(self,
               adversary,
               steps=10,
               verbose=False,
               dispersion_type=None):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            steps: int. The number of steps to find adversary example.
            verbose: bool. log attack process if true.

        Returns:
            Adversary instance with possible changed status.
        """
        assert dispersion_type in self.support_type, self.support_type
        norm = self.norm
        epsilon_ball = self.epsilon_ball
        # Unused
        # epsilon_stepsize = self.epsilon_stepsize

        original_img = adversary.denormalized_original

        if adversary.is_targeted_attack:
            raise ValueError("This attack method only support untargeted attack!")

        box_constrains_lower_bound = self.box_constrains_lower_bound
        box_constrains_upper_bound = self.box_constrains_upper_bound

        adv_img = original_img + 0.001 * np.random.standard_normal(original_img.shape)
        adv_img = paddle.to_tensor(adv_img, dtype='float32', place=self._device)
        original_img = paddle.to_tensor(original_img, dtype='float32', place=self._device)
        original_img_normalized = self.normalize(paddle.squeeze(original_img))
        if len(original_img_normalized.shape) < 4:
            original_img_normalized = paddle.unsqueeze(original_img_normalized, axis=0)
        logits = self.model.predict_tensor(original_img_normalized)

        if dispersion_type == self.support_type[0]:
            if norm == 'Linf':
                step_size = epsilon_ball / steps
                for _ in range(steps):
                    adv_img.stop_gradient = False
                    adv_img_normalized = self.normalize(paddle.squeeze(adv_img))
                    if len(adv_img_normalized.shape) < 4:
                        adv_img_normalized = paddle.unsqueeze(adv_img_normalized, axis=0)

                    logits_advs = self.model.predict_tensor(adv_img_normalized)
                    loss_logits_kl = self.kldiv_criterion(self.logsoftmax(logits_advs), self.softmax(logits))

                    grad = paddle.autograd.grad(loss_logits_kl, adv_img)[0]

                    # avoid nan or inf if gradient is 0
                    if grad.isnan().any():
                        paddle.assign(0.001 * paddle.randn(grad.shape), grad)
                    # TODO: fix nan error
                    adv_img = adv_img.detach() + step_size * paddle.sign(grad.detach())
                    eta = paddle.clip(adv_img - original_img, - epsilon_ball, epsilon_ball)
                    adv_img = original_img + eta
                    adv_img = paddle.clip(adv_img, box_constrains_lower_bound, box_constrains_upper_bound)
                    # if grad.isnan().any() or adv_img.isnan().any():
                    #     import pdb
                    #     pdb.set_trace()
                    # else:
                    #     pass
                    '''
                    loss_logits_kl = self.kldiv_criterion(self.logsoftmax(logits_advs), self.softmax(logits))
                    loss_logits_kl.backward()
                    grad = adv_img.grad
                    adv_img = adv_img.detach() + step_size * paddle.sign(grad.detach())
                    # adv_img = paddle.min(paddle.max(adv_img, original_img - epsilon_ball), original_img + epsilon_ball)
                    adv_img = paddle.clip(adv_img, original_img - epsilon_ball, original_img + epsilon_ball)
                    adv_img = paddle.clip(adv_img, box_constrains_lower_bound, box_constrains_upper_bound)
                    '''
            elif norm == 'L2':
                delta = 0.001 * paddle.randn(original_img.shape).detach()
                # Setup optimizers
                # lr = epsilon_ball / steps * 2, because of the (v - epsilon_ball, v + epsilon_ball) limit within steps.
                optimizer_delta = paddle.optimizer.SGD(learning_rate=epsilon_ball / steps * 2, parameters=[delta])

                for _ in range(steps):
                    delta.stop_gradient = False
                    adv_img = original_img + delta
                    # optimize
                    optimizer_delta.clear_grad()

                    adv_img_normalized = self.normalize(paddle.squeeze(adv_img))
                    if len(adv_img_normalized.shape) < 4:
                        adv_img_normalized = paddle.unsqueeze(adv_img_normalized, axis=0)

                    logits_advs = self.model.predict_tensor(adv_img_normalized)
                    loss_logits_kl = (-1) * self.kldiv_criterion(self.logsoftmax(logits_advs), self.softmax(logits))
                    loss_logits_kl.backward()

                    # renorming gradient
                    grad_norms = paddle.norm(delta.grad, p=2)
                    delta.grad.divide(grad_norms)

                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any() or grad_norms.isnan():
                        paddle.assign(0.0001 * paddle.randn(delta.grad.shape), delta.grad)

                    optimizer_delta.step()

                    # projection
                    delta.add(original_img)
                    delta.clip(box_constrains_lower_bound, box_constrains_upper_bound).add(-original_img)
                    delta_norm = paddle.norm(delta, p=2)
                    delta.divide(delta_norm)
                    # clip epsilon_ball for safety.
                    delta.clip(-epsilon_ball, epsilon_ball)

                adv_img = original_img.detach() + delta.detach()

        # TODO: ALP L2 logits dispersion
        elif dispersion_type == self.support_type[1]:
            print("developing")
            exit(0)
        # TODO: DLR in Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks
        elif dispersion_type == self.support_type[2]:
            print("developing")
            exit(0)
        else:
            print("developing")
            exit(0)
            adv_img = np.clip(original_img, box_constrains_lower_bound, box_constrains_upper_bound)

        adv_label = np.argmax(self.model.predict(adv_img_normalized))
        adversary.try_accept_the_example(np.squeeze(adv_img.numpy()),
                                         np.squeeze(adv_img_normalized.numpy()),
                                         adv_label)
        return adversary


LD = LOGITS_DISPERSION
