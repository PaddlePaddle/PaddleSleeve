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
    # TODO: seperate as three: softmax kl, logits norm, DLR
    * softmax_kl, softmax of logits KL dispersion
    * logits_norm, logits of norm distance dispersion, for instantce ||logit - logits_adv||22
    * difference_logits_ratio, Difference of Logits Ratio Loss. DLR(x, y) = − (zy − max(i!=y)zi) / (zπ1 − zπ3)
    Args:
        model: PaddleWhiteBoxModel.
        learning_rate: float. for adam optimizer.
    """
    def __init__(self, model, norm='Linf', epsilon_ball=8/255, dispersion_type=None):
        super(LOGITS_DISPERSION, self).__init__(model, norm=norm, epsilon_ball=epsilon_ball)
        self.model = model
        self.safe_num = 0.999999
        # (float, float), It is used to map input float into (0, 1) for arctanh
        self.box_constrains_lower_bound = self.model.bounds[0]
        self.box_constrains_upper_bound = self.model.bounds[1]
        assert self.box_constrains_lower_bound < self.box_constrains_upper_bound

        self.support_type = ('softmax_kl', 'logits_norm', 'difference_logits_ratio')
        assert dispersion_type in self.support_type
        self.dispersion_type = dispersion_type
        self.kldiv_criterion = paddle.nn.KLDivLoss(reduction='batchmean')
        self.logsoftmax = paddle.nn.LogSoftmax()
        self.softmax = paddle.nn.Softmax()

    def _apply(self,
               adversary,
               perturb_steps=10,
               verbose=False,
               ):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            perturb_steps: int. The number of steps to find adversary example.
            verbose: bool. log attack process if true.

        Returns:
            Adversary instance with possible changed status.
        """
        epsilon = self.epsilon_ball
        original_img = adversary.original
        if len(original_img.shape) < 4:
            original_img = np.expand_dims(original_img, axis=0)

        if adversary.is_targeted_attack:
            raise ValueError("This attack method only support untargeted attack!")

        box_constrains_lower_bound = self.box_constrains_lower_bound
        box_constrains_upper_bound = self.box_constrains_upper_bound

        adv_img = original_img + 0.001 * np.random.standard_normal(original_img.shape)
        original_img = paddle.to_tensor(original_img, dtype='float32', place=self._device)
        adv_img = paddle.to_tensor(adv_img, dtype='float32', place=self._device)
        dispersion_type = self.dispersion_type
        if dispersion_type == self.support_type[0]:
            if self.norm == 'Linf':
                step_size = epsilon / perturb_steps
                for _ in range(perturb_steps):
                    adv_img.stop_gradient = False
                    logits = self.model.predict_tensor(original_img)
                    logits_advs = self.model.predict_tensor(adv_img)
                    loss_logits_kl = self.kldiv_criterion(self.logsoftmax(logits_advs), self.softmax(logits))

                    grad = paddle.autograd.grad(loss_logits_kl, adv_img)[0]
                    adv_img = adv_img.detach() + step_size * paddle.sign(grad.detach())
                    adv_img = paddle.clip(adv_img, original_img - epsilon, original_img + epsilon)
                    adv_img = paddle.clip(adv_img, box_constrains_lower_bound, box_constrains_upper_bound)
                    '''
                    loss_logits_kl = self.kldiv_criterion(self.logsoftmax(logits_advs), self.softmax(logits))
                    loss_logits_kl.backward()
                    grad = adv_img.grad
                    adv_img = adv_img.detach() + step_size * paddle.sign(grad.detach())
                    # adv_img = paddle.min(paddle.max(adv_img, original_img - epsilon), original_img + epsilon)
                    adv_img = paddle.clip(adv_img, original_img - epsilon, original_img + epsilon)
                    adv_img = paddle.clip(adv_img, box_constrains_lower_bound, box_constrains_upper_bound)
                    '''
                # import pdb
                # pdb.set_trace()
            elif self.norm == 'L2':
                # TODO: finsh this.
                delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
                delta = Variable(delta.data, requires_grad=True)
                # Setup optimizers
                # lr = epsilon / perturb_steps * 2, because of the (v - epsilon, v + epsilon) limit within perturb_steps.
                optimizer_delta = paddle.optimizer.SGD(learning_rate=epsilon / perturb_steps * 2, parameters=[delta])

                for _ in range(perturb_steps):
                    adv = x_natural + delta

                    # optimize
                    optimizer_delta.zero_grad()
                    with torch.enable_grad():
                        loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                                   F.softmax(model(x_natural), dim=1))
                    loss.backward()
                    # renorming gradient
                    grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                    delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                    optimizer_delta.step()

                    # projection
                    delta.data.add_(x_natural)
                    delta.data.clamp_(0, 1).sub_(x_natural)
                    delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
                x_adv = Variable(x_natural + delta, requires_grad=False)
        # TODO: ALP L2 logits dispersion
        elif dispersion_type == self.support_type[1]:
            pass
        # TODO: DLR in Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks
        elif dispersion_type == self.support_type[2]:
            pass
        else:
            x_adv = np.clip(original_img, box_constrains_lower_bound, box_constrains_upper_bound)

        adv_label = np.argmax(self.model.predict(paddle.to_tensor(adv_img)))
        adversary.try_accept_the_example(np.squeeze(adv_img.numpy()), adv_label)
        return adversary


LD = LOGITS_DISPERSION
