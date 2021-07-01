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

import logging
from collections import Iterable

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
    This class implements gradient attack method, and is the base of FGSM, BIM, ILCM, etc.
    """
    def __init__(self, model, support_targeted=True, pgd_flag=False):
        """
        Args:
            model: An instance of a paddle model to be attacked.
            support_targeted(Does): this attack method support targeted.
            pgd_flag: place it true if use pgd
        """
        super(GradientMethodAttack, self).__init__(model)
        self.support_targeted = support_targeted
        self.pgd_flag = pgd_flag

    def _apply(self,
               adversary,
               norm_ord=None,
               epsilons=0.01,
               epsilon_steps=10,
               steps=100,
               perturb=16.0 / 256,
               ):
        """
        Apply the gradient attack method.
        Args:
            adversary: The Adversary object.
            norm_ord: Order of the norm, such as np.inf, 1, 2, etc. It can't be 0.
            epsilons: Attack step size (input variation). Largest step size if epsilons is not iterable.
            epsilon_steps: The number of Epsilons' iteration for each attack iteration.
            steps: The number of attack iteration.

        Returns:
            adversary(Adversary): The Adversary object.
        """
        if norm_ord is None:
            norm_ord = np.inf

        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        if not isinstance(epsilons, Iterable):
            if epsilon_steps == 1:
                epsilons = [epsilons]
            else:
                epsilons = np.linspace(0, epsilons, num=epsilon_steps)
        assert self.model.channel_axis == adversary.original.ndim
        assert (self.model.channel_axis == 1 or
                self.model.channel_axis == adversary.original.shape[0] or
                self.model.channel_axis == adversary.original.shape[-1])

        original_label = adversary.original_label
        min_, max_ = self.model.bounds
        adv_img = adversary.original
        if len(adv_img.shape) < 4:
            adv_img = np.expand_dims(adv_img, axis=0)

        adv_img = paddle.to_tensor(adv_img, dtype='float32', place=self._device)
        adv_img.stop_gradient = False

        if adversary.is_targeted_attack:
            target_label = adversary.target_label
            target_label = paddle.to_tensor(target_label, dtype='int64', place=self._device)
        for epsilon in epsilons[:]:
            if epsilon == 0.0:
                continue

            for step in range(steps):
                if adversary.is_targeted_attack:
                    gradient = - self.model.gradient(adv_img, target_label)
                else:
                    gradient = self.model.gradient(adv_img, original_label)

                gradient = paddle.to_tensor(gradient, dtype='float32', place=self._device)
                if norm_ord == np.inf:
                    gradient_norm = paddle.sign(gradient)
                else:
                    gradient_norm = gradient / self._norm(gradient.numpy(), ord=norm_ord)

                if len(adv_img.shape) < 4:
                    adv_img = np.expand_dims(adv_img.numpy(), axis=0)

                if self.pgd_flag:
                    # linf
                    adv_img = adv_img + gradient_norm * epsilon
                    clip_max = np.clip(adv_img.numpy() * (1.0 + perturb), min_, max_)
                    clip_min = np.clip(adv_img.numpy() * (1.0 - perturb), min_, max_)
                    adv_img = np.clip(adv_img.numpy(), clip_min, clip_max)  
                    adv_label = np.argmax(self.model.predict(paddle.to_tensor(adv_img)))
                    adv_img = paddle.to_tensor(adv_img)
                else:
                    adv_img = adv_img + gradient_norm * epsilon
                    adv_label = np.argmax(self.model.predict(adv_img))

                if adversary.try_accept_the_example(np.squeeze(adv_img.numpy()), adv_label):
                    return adversary

        return adversary

    @staticmethod
    def _norm(a, ord):
        if a.ndim == 1 or a.ndim == 2:
            return np.linalg.norm(a, ord=ord)
        # channel first
        elif a.ndim == a.shape[0]:
            norm_shape = a.ndim * a.shape[1:][0] * a.shape[1:][0]
        # channel last
        else:
            norm_shape = a.ndim * a.shape[:-1][0] * a.shape[:-1][1]
        return np.linalg.norm(a.reshape(norm_shape), ord=ord)


class FastGradientSignMethodTargetedAttack(GradientMethodAttack):
    """
    "Fast Gradient Sign Method" is extended to support targeted attack.
    "Fast Gradient Sign Method" was originally implemented by Goodfellow et
    al. (2015) with the infinity norm.

    Paper link: https://arxiv.org/abs/1412.6572
    """

    def _apply(self, adversary, **kwargs):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            **kwargs: Other named arguments.

        Returns:
            An adversary status with changed status.
        """

        return GradientMethodAttack._apply(self, adversary=adversary, **kwargs)


class FastGradientSignMethodAttack(FastGradientSignMethodTargetedAttack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm, and is known as the "Fast Gradient Sign Method".

    Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model):
        """
        FGSM attack init.
        Args:
            model: PaddleWhiteBoxModel.
        """

        super(FastGradientSignMethodAttack, self).__init__(model, False)


class IterativeLeastLikelyClassMethodAttack(GradientMethodAttack):
    """
    "Iterative Least-likely Class Method (ILCM)" extends "BIM" to support
    targeted attack.
    "The Basic Iterative Method (BIM)" is to extend "FSGM". "BIM" iteratively
    take multiple small steps while adjusting the direction after each step.

    Paper link: https://arxiv.org/abs/1607.02533
    """

    def _apply(self, adversary, epsilons=0.01, steps=1000):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            epsilons: float. A single step perturbation length.
            steps: int. Total steps number.

        Returns:
            An adversary status with changed status.
        """
        return GradientMethodAttack._apply(
            self,
            adversary=adversary,
            norm_ord=np.inf,
            epsilons=epsilons,
            steps=steps)


class BasicIterativeMethodAttack(IterativeLeastLikelyClassMethodAttack):
    """
    FGSM is a one-step method. "The Basic Iterative Method (BIM)" iteratively
    take multiple small steps while adjusting the direction after each step.
    Paper link: https://arxiv.org/abs/1607.02533
    """

    def __init__(self, model):
        """

        Args:
            model: PaddleWhiteBoxModel.
        """
        super(BasicIterativeMethodAttack, self).__init__(model, False)


class MomentumIteratorAttack(GradientMethodAttack):
    """
    The Momentum Iterative Fast Gradient Sign Method (Dong et al. 2017).
    This method won the first places in NIPS 2017 Non-targeted Adversarial
    Attacks and Targeted Adversarial Attacks. The original paper used
    hard labels for this attack; no label smoothing. inf norm.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(self, model, support_targeted=True):
        """
        MIFGSM attack init.
        Args:
            model: PaddleWhiteBoxModel.
            support_targeted: bool.
        """
        super(MomentumIteratorAttack, self).__init__(model)
        self.support_targeted = support_targeted

    def _apply(self,
               adversary,
               norm_ord=None,
               epsilons=0.1,
               steps=100,
               epsilon_steps=100,
               decay_factor=1):
        """
        Apply the momentum iterative gradient attack method.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            norm_ord: int. Order of the norm, such as np.inf, 1, 2, etc. It can't be 0.
            epsilons: (list|tuple|float). Attack step size (input variation). Largest step size if epsilons is not iterable.
            steps: int. The number of attack iteration.
            epsilon_steps: int. The number of Epsilons' iteration for each attack iteration.
            decay_factor: float. The decay factor for the momentum term.

        Returns:
            An adversary status with changed status.
        """
        if norm_ord is None:
            norm_ord = np.inf

        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, epsilons, num=epsilon_steps)

        min_, max_ = self.model.bounds

        original_label = adversary.original_label
        original_label = paddle.to_tensor(original_label, dtype='int64', place=self._device)

        if adversary.is_targeted_attack:
            target_label = adversary.target_label
            target_label = paddle.to_tensor(target_label, dtype='int64', place=self._device)

        for epsilon in epsilons[:]:
            if epsilon == 0.0:
                continue

            adv_img = adversary.original
            if len(adv_img.shape) < 4:
                adv_img = np.expand_dims(adv_img, axis=0)
            adv_img = paddle.to_tensor(adv_img, dtype='float32', place=self._device)
            adv_img.stop_gradient = False

            momentum = 0
            for step in range(steps):

                if adversary.is_targeted_attack:
                    gradient = - self.model.gradient(adv_img, target_label)
                else:
                    gradient = self.model.gradient(adv_img, original_label)

                gradient = np.squeeze(gradient)
                velocity = gradient / self._norm(gradient, ord=1)
                velocity = np.expand_dims(velocity, axis=0)

                momentum = decay_factor * momentum + velocity
                if norm_ord == np.inf:
                    normalized_grad = np.sign(momentum)
                else:
                    normalized_grad = self._norm(momentum, ord=norm_ord)

                perturbation = epsilon * normalized_grad
                perturbation = paddle.to_tensor(perturbation)
                adv_img = adv_img + perturbation
                adv_label = np.argmax(self.model.predict(adv_img))

                logging.info('step={}, epsilon = {:.5f}, pre_label = {}, adv_label={}' .format(step,
                                                                                               epsilon,
                                                                                               original_label,
                                                                                               adv_label))

                if adversary.try_accept_the_example(np.squeeze(adv_img.numpy()), adv_label):
                    return adversary

        return adversary


class ProjectedGradientDescentAttack(GradientMethodAttack):
    """
    Projected Gradient Descent
    Towards deep learning models resistant to adversarial attacks, A. Madry, A. Makelov, L. Schmidt, D. Tsipras, 
    and A. Vladu, ICLR 2018
    """

    def __init__(self, model, support_targeted=True, pgd_flag=True):
        """
        PGD attack init.
        Args:
            model: PaddleWhiteBoxModel.
        """
        super(ProjectedGradientDescentAttack, self).__init__(model)
        self.support_targeted = support_targeted
        self.pgd_flag = pgd_flag 


FGSM = FastGradientSignMethodAttack
FGSMT = FastGradientSignMethodTargetedAttack
BIM = BasicIterativeMethodAttack
ILCM = IterativeLeastLikelyClassMethodAttack
MIFGSM = MomentumIteratorAttack
PGD = ProjectedGradientDescentAttack
