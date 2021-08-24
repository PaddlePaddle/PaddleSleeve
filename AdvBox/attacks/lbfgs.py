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
This module provides the attack method of "LBFGS".
"""
from __future__ import division

from builtins import range
import logging

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .base import Attack
import paddle

__all__ = ['LBFGSAttack', 'LBFGS']


class LBFGSAttack(Attack):
    """
    Uses L-BFGS-B to minimize the cross-entropy and the distance between the
    original and the adversary.
    Paper link: https://arxiv.org/abs/1510.05328
    """

    def __init__(self, model, norm='L2', epsilon_ball=8/255, epsilon_stepsize=2/255):
        super(LBFGSAttack, self).__init__(model,
                                          norm=norm,
                                          epsilon_ball=epsilon_ball,
                                          epsilon_stepsize=epsilon_stepsize)
        self._predicts_normalized = None

    def _apply(self, adversary, confidence=0.01, steps=10):
        if not adversary.is_targeted_attack:
            raise ValueError("This attack method only support targeted attack!")

        # finding initial c
        logging.info('finding initial c...')
        confidence_current = confidence
        x0 = np.copy(adversary.denormalized_original.flatten())
        for i in range(30):
            confidence_current = 2 * confidence_current
            logging.info('c={}'.format(confidence_current))
            is_adversary = self._lbfgsb(adversary, x0, confidence_current, steps)
            if is_adversary:
                break
        if not is_adversary:
            logging.info('Failed!')
            return adversary

        # binary search c
        logging.info('binary search c...')
        c_low = 0
        c_high = confidence_current
        while c_high - c_low >= confidence:
            logging.info('c_high={}, c_low={}, diff={}, epsilon={}'
                         .format(c_high, c_low, c_high - c_low, confidence))
            c_half = (c_low + c_high) / 2
            is_adversary = self._lbfgsb(adversary, x0, c_half, steps)
            if is_adversary:
                c_high = c_half
            else:
                c_low = c_half

        return adversary

    #def _is_predicts_normalized(self, predicts):
    #    """
    #    To determine the predicts is normalized.
    #    :param predicts(np.array): the output of the model.
    #    :return: bool
    #    """
    #    if self._predicts_normalized is None:
    #        if self.model.predict_name().lower() in [
    #                'softmax', 'probabilities', 'probs'
    #        ]:
    #            self._predicts_normalized = True
    #        else:
    #            if np.any(predicts < 0.0):
    #                self._predicts_normalized = False
    #            else:
    #                s = np.sum(predicts.flatten())
    #                if 0.999 <= s <= 1.001:
    #                    self._predicts_normalized = True
    #                else:
    #                    self._predicts_normalized = False
    #    assert self._predicts_normalized is not None
    #    return self._predicts_normalized

    def _loss(self, adv_img, confidence, adversary):
        """
        To get the loss and gradient.
        :param adv_x: the candidate adversarial example
        :param c: parameter 'C' in the paper
        :return: (loss, gradient)
        """
        adv_img_reshaped = adv_img.reshape(adversary.original.shape)
        # x = adv_img.reshape(adversary.original.shape)
        # img = adv_img.reshape([1] + [v for v in adversary.original.shape])
        adv_img_reshaped_tensor = paddle.to_tensor(adv_img_reshaped, dtype='float32', place=self._device)
        adv_img_reshaped_tensor.stop_gradient = False
        adv_img_reshaped_tensor_normalized = self.input_preprocess(adv_img_reshaped_tensor)

        # numpy computation
        logits_np = self.model.predict(adv_img_reshaped_tensor_normalized.numpy())
        e = np.exp(logits_np)
        logits_np = e / np.sum(e)
        e = np.exp(logits_np)
        s = np.sum(e)
        ce = np.log(s) - logits_np[0, adversary.target_label]

        min_, max_ = self.model.bounds
        if self.norm == 'L2':
            d = np.sum((adv_img_reshaped - adversary.denormalized_original).flatten() ** 2) \
                / ((max_ - min_) ** 2) / len(adv_img)
        elif self.norm == 'Linf':
            # TODO: add Linf distance attack
            exit(1)
        else:
            exit(1)

        # gradient
        logits_tensor = self.model.predict_tensor(adv_img_reshaped_tensor_normalized)
        target_label = paddle.to_tensor(adversary.target_label, dtype='int64', place=self._device)
        loss = self.model.loss(logits_tensor, target_label)
        loss.backward(retain_graph=True)
        gradient = adv_img_reshaped_tensor.grad.numpy()
        # gradient = self.model.gradient(img_normalized, adversary.target_label)
        result = (confidence * ce + d).astype(float), gradient.flatten().astype(float)
        return result

    def _lbfgsb(self, adversary, img0, confidence, maxiter):
        min_, max_ = self.model.bounds
        bounds = [(min_, max_)] * len(img0)
        approx_grad_eps = (max_ - min_) / 100.0

        adv_img, f, d = fmin_l_bfgs_b(self._loss, img0, args=(confidence, adversary, ), bounds=bounds, maxiter=maxiter, epsilon=approx_grad_eps)
        if np.amax(adv_img) > max_ or np.amin(adv_img) < min_:
            adv_img = np.clip(adv_img, min_, max_)

        # TODO：use epsilon_ball and epsilon_stepsize control
        shape = adversary.original.shape
        adv_img_reshaped = adv_img.reshape(shape)
        adv_img_tensor = paddle.to_tensor(adv_img_reshaped, dtype='float32', place=self._device)
        adv_img_reshaped_tensor_normalized = self.input_preprocess(adv_img_tensor)
        adv_label = np.argmax(self.model.predict(adv_img_reshaped_tensor_normalized))
        logging.info('pre_label = {}, adv_label={}'.format(adversary.target_label, adv_label))

        adv_img_tensor = self.safe_delete_batchsize_dimension(adv_img_tensor)
        adv_img_normalized = self.safe_delete_batchsize_dimension(adv_img_reshaped_tensor_normalized)
        is_ok = adversary.try_accept_the_example(adv_img_tensor.numpy(),
                                                 adv_img_normalized.numpy(),
                                                 adv_label)

        return is_ok


LBFGS = LBFGSAttack
