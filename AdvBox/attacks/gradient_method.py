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
This module provides the implementation for FGSM attack method.
"""
from __future__ import division
import numpy as np
import paddle
from .base import Attack


__all__ = [
    'GradientMethodAttack', 'FastGradientSignMethodAttack', 'FGSM',
    'FastGradientSignMethodTargetedAttack', 'FGSMT',
    'BasicIterativeMethodAttack', 'BIM',
    'IterativeLeastLikelyClassMethodAttack', 'ILCM',
    'MomentumIteratorAttack', 'MIFGSM',
    'ProjectedGradientDescentAttack', 'PGD'
]


class GradientMethodAttack(Attack):
    """
    This class implements gradient attack method, and is the base of FGSM, BIM, ILCM.
    """
    def __init__(self, model, norm='Linf',
                 epsilon_ball=8/255, epsilon_stepsize=2/255,
                 support_targeted=True):
        """
        Args:
            model: An instance of a paddle model to be attacked.
            support_targeted(Does): this attack method support targeted.
        """
        super(GradientMethodAttack, self).__init__(model, norm=norm,
                                                   epsilon_ball=epsilon_ball,
                                                   epsilon_stepsize=epsilon_stepsize)
        self.support_targeted = support_targeted

    def _apply(self,
               adversary,
               steps=20,
               stop_early=False):
        """
        Apply the gradient attack method.
        Args:
            adversary: The Adversary object.
            steps: The number of attack iteration.
            stop_early: bool. iteration attack will stop early if true.

        Returns:
            adversary(Adversary): The Adversary object.
        """
        norm = self.norm
        epsilon_ball = self.epsilon_ball
        epsilon_stepsize = self.epsilon_stepsize

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError("This attack method doesn't support targeted attack!")

        original_label = adversary.original_label
        original_label = paddle.to_tensor(original_label, dtype='int64', place=self._device)
        min_, max_ = self.model.bounds

        if adversary.is_targeted_attack:
            target_label = adversary.target_label
            num_labels = self.model.num_classes()
            assert target_label < num_labels
            target_label = paddle.to_tensor(target_label, dtype='int64', place=self._device)

        img = adversary.denormalized_original
        img_tensor = paddle.to_tensor(img, dtype='float32', place=self._device)
        adv_img = paddle.to_tensor(img, dtype='float32', place=self._device)
        for step in range(steps):
            adv_img.stop_gradient = False
            adv_img_normalized = self.input_preprocess(adv_img)

            if adversary.is_targeted_attack:
                logits = self.model.predict_tensor(adv_img_normalized)
                loss = self.model.loss(logits, target_label)
                loss.backward(retain_graph=True)
                gradient = - adv_img.grad
            else:
                logits = self.model.predict_tensor(adv_img_normalized)
                # wangwenhua add
                if original_label.ndim == 0:
                    original_label = original_label.reshape([1])
                loss = self.model.loss(logits, original_label)
                loss.backward(retain_graph=True)
                gradient = adv_img.grad

            if norm == 'Linf':
                normalized_gradient = paddle.sign(gradient)
            elif norm == 'L2':
                gradient_norm = paddle.norm(gradient, p=2)
                normalized_gradient = gradient / gradient_norm
            else:
                exit(1)

            # control norm and clip in model.bounds domain.
            eta = epsilon_stepsize * normalized_gradient
            adv_img = adv_img.detach() + eta.detach()
            eta = paddle.clip(adv_img - img_tensor, -epsilon_ball, epsilon_ball)
            adv_img = img_tensor + eta
            adv_img = paddle.clip(adv_img, min_, max_).detach()

            adv_img_normalized = self.input_preprocess(adv_img)
            adv_label = np.argmax(self.model.predict(adv_img_normalized))
            adv_img = self.safe_delete_batchsize_dimension(adv_img)
            adv_img_normalized = self.safe_delete_batchsize_dimension(adv_img_normalized)

            if stop_early:
                if adversary.try_accept_the_example(adv_img.numpy(),
                                                    adv_img_normalized.numpy(),
                                                    adv_label):
                    return adversary

        adversary.try_accept_the_example(adv_img.numpy(),
                                         adv_img_normalized.numpy(),
                                         adv_label)
        return adversary


class FastGradientSignMethodTargetedAttack(GradientMethodAttack):
    """
    "Fast Gradient Sign Method" is extended to support targeted attack.
    "Fast Gradient Sign Method" was originally implemented by Goodfellow et
    al. (2015) with the infinity norm.

    Paper link: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, model, norm='Linf', epsilon_ball=8/255, epsilon_stepsize=2/255):
        """
        FGSM attack init.
        Args:
            model: PaddleWhiteBoxModel.
        """
        super(FastGradientSignMethodTargetedAttack, self).__init__(model,
                                                                   norm=norm,
                                                                   epsilon_ball=epsilon_ball,
                                                                   epsilon_stepsize=epsilon_stepsize,
                                                                   support_targeted=True)

    def _apply(self, adversary, **kwargs):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            **kwargs: Other named arguments.

        Returns:
            An adversary status with changed status.
        """
        return GradientMethodAttack._apply(self,
                                           adversary=adversary,
                                           steps=1)


class FastGradientSignMethodAttack(GradientMethodAttack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm, and is known as the "Fast Gradient Sign Method".

    Paper link: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, model, norm='Linf', epsilon_ball=8/255, epsilon_stepsize=2/255):
        """
        FGSM attack init.
        Args:
            model: PaddleWhiteBoxModel.
        """
        super(FastGradientSignMethodAttack, self).__init__(model,
                                                           norm=norm,
                                                           epsilon_ball=epsilon_ball,
                                                           epsilon_stepsize=epsilon_stepsize,
                                                           support_targeted=False)

    def _apply(self, adversary, **kwargs):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            **kwargs: Other named arguments.

        Returns:
            An adversary status with changed status.
        """
        return GradientMethodAttack._apply(self,
                                           adversary=adversary,
                                           steps=1)


class ProjectedGradientDescentAttack(GradientMethodAttack):
    """
    Projected Gradient Descent
    Towards deep learning models resistant to adversarial attacks, A. Madry, A. Makelov, L. Schmidt, D. Tsipras,
    and A. Vladu, ICLR 2018
    """
    def __init__(self, model, norm='Linf', epsilon_ball=8/255, epsilon_stepsize=2/255):
        """
        PGD attack init.
        Args:
            model: PaddleWhiteBoxModel.
        """
        super(ProjectedGradientDescentAttack, self).__init__(model,
                                                             norm=norm,
                                                             epsilon_ball=epsilon_ball,
                                                             epsilon_stepsize=epsilon_stepsize,
                                                             support_targeted=True)

    def _apply(self, adversary, **kwargs):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            **kwargs: Other named arguments.

        Returns:
            An adversary status with changed status.
        """
        return GradientMethodAttack._apply(self,
                                           adversary=adversary,
                                           **kwargs)


class IterativeLeastLikelyClassMethodAttack(GradientMethodAttack):
    """
    "Iterative Least-likely Class Method (ILCM)" extends "BIM" to support
    targeted attack.
    "The Basic Iterative Method (BIM)" is to extend "FSGM". "BIM" iteratively
    take multiple small steps while adjusting the direction after each step.

    Paper link: https://arxiv.org/abs/1607.02533
    """

    def _apply(self, adversary, steps=1000):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            epsilons: float. A single step perturbation length.
            steps: int. Total steps number.

        Returns:
            An adversary status with changed status.
        """
        return GradientMethodAttack._apply(self,
                                           adversary=adversary,
                                           steps=steps)


class BasicIterativeMethodAttack(IterativeLeastLikelyClassMethodAttack):
    """
    FGSM is a one-step method. "The Basic Iterative Method (BIM)" iteratively
    take multiple small steps while adjusting the direction after each step.
    Paper link: https://arxiv.org/abs/1607.02533
    """
    def __init__(self, model, norm='Linf', epsilon_ball=8/255, epsilon_stepsize=2/255):
        """

        Args:
            model: PaddleWhiteBoxModel.
        """
        super(BasicIterativeMethodAttack, self).__init__(model, norm=norm,
                                                         epsilon_ball=epsilon_ball,
                                                         epsilon_stepsize=epsilon_stepsize,
                                                         support_targeted=False)


class MomentumIteratorAttack(Attack):
    """
    The Momentum Iterative Fast Gradient Sign Method (Dong et al. 2017).
    This method won the first places in NIPS 2017 Non-targeted Adversarial
    Attacks and Targeted Adversarial Attacks. The original paper used
    hard labels for this attack; no label smoothing. inf norm.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """
    def __init__(self, model, norm='Linf', epsilon_ball=100/255, epsilon_stepsize=2/255):
        """
        MIFGSM attack init.
        Args:
            model: PaddleWhiteBoxModel.
        """
        super(MomentumIteratorAttack, self).__init__(model,
                                                     norm=norm,
                                                     epsilon_ball=epsilon_ball,
                                                     epsilon_stepsize=epsilon_stepsize)

    def _apply(self,
               adversary,
               steps=100,
               decay_factor=1,
               stop_early=False):
        """
        Apply the momentum iterative gradient attack method.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            steps: int. The number of attack iteration.
            decay_factor: float. The decay factor for the momentum term.

        Returns:
            An adversary status with changed status.
        """
        norm = self.norm
        epsilon_ball = self.epsilon_ball
        epsilon_stepsize = self.epsilon_stepsize

        min_, max_ = self.model.bounds

        original_label = adversary.original_label
        original_label = paddle.to_tensor(original_label, dtype='int64', place=self._device)

        if adversary.is_targeted_attack:
            target_label = adversary.target_label
            num_labels = self.model.num_classes()
            assert target_label < num_labels
            target_label = paddle.to_tensor(target_label, dtype='int64', place=self._device)

        img = adversary.denormalized_original
        img_tensor = paddle.to_tensor(img, dtype='float32', place=self._device)
        adv_img = paddle.to_tensor(img, dtype='float32', place=self._device)

        momentum = 0
        for step in range(steps):
            adv_img.stop_gradient = False
            adv_img_normalized = self.input_preprocess(adv_img)

            if adversary.is_targeted_attack:
                logits = self.model.predict_tensor(adv_img_normalized)
                loss = self.model.loss(logits, target_label)
                loss.backward(retain_graph=True)
                gradient = - adv_img.grad

            else:
                logits = self.model.predict_tensor(adv_img_normalized)
                loss = self.model.loss(logits, original_label)
                loss.backward(retain_graph=True)
                gradient = adv_img.grad

            gradient_norm = paddle.norm(gradient, p=1)
            velocity = gradient / gradient_norm
            momentum = decay_factor * momentum + velocity
            if norm == 'Linf':
                normalized_momentum = paddle.sign(momentum)
            elif norm == 'L2':
                momentum_norm = paddle.norm(momentum, p=2)
                normalized_momentum = momentum / momentum_norm
            else:
                exit(1)

            eta = epsilon_stepsize * normalized_momentum
            adv_img = adv_img.detach() + eta.detach()
            eta = paddle.clip(adv_img - img_tensor, -epsilon_ball, epsilon_ball)
            adv_img = img_tensor + eta
            adv_img = paddle.clip(adv_img, min_, max_).detach()

            adv_img_normalized = self.input_preprocess(adv_img)
            adv_label = np.argmax(self.model.predict(adv_img_normalized))
            adv_img = self.safe_delete_batchsize_dimension(adv_img)
            adv_img_normalized = self.safe_delete_batchsize_dimension(adv_img_normalized)
            if stop_early:
                if adversary.try_accept_the_example(adv_img.numpy(),
                                                    adv_img_normalized.numpy(),
                                                    adv_label):
                    return adversary
        adversary.try_accept_the_example(adv_img.numpy(),
                                         adv_img_normalized.numpy(),
                                         adv_label)
        return adversary


FGSM = FastGradientSignMethodAttack
FGSMT = FastGradientSignMethodTargetedAttack
BIM = BasicIterativeMethodAttack
ILCM = IterativeLeastLikelyClassMethodAttack
MIFGSM = MomentumIteratorAttack
PGD = ProjectedGradientDescentAttack
