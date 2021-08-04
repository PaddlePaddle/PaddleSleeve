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
This module provides the attack method of "CW".
L2 distance metrics especially
"""
from __future__ import division
from __future__ import print_function

from builtins import range
import numpy as np
import paddle
from .base import Attack


__all__ = ['CWL2Attack', 'CW_L2']


class CWL2Attack(Attack):
    """
    Uses Adam to minimize the CW L2 objective function
    Paper link: https://arxiv.org/abs/1608.04644
    Args:
        model: PaddleWhiteBoxModel.
        learning_rate: float. for adam optimizer.
    """
    def __init__(self, model, norm='L2', epsilon_ball=8/255, epsilon_stepsize=2/255):
        super(CWL2Attack, self).__init__(model,
                                         norm=norm,
                                         epsilon_ball=epsilon_ball,
                                         epsilon_stepsize=epsilon_stepsize)
        assert norm == 'L2', "Only support L2 CW for now."
        self.model = model
        self.safe_num = 0.999999
        # (float, float), It is used to map input float into (0, 1) for arctanh
        self.sample_float_range = self.model.bounds
        assert self.sample_float_range[0] < self.sample_float_range[1]

    def _apply(self,
               adversary,
               attack_iterations=10,
               c_search_steps=10,
               c_range=(0.01, 100),
               c_accuracy=0.01,
               k_threshold=0,
               verbose=True):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            attack_iterations: int. optimization step using adam for perturbation.
            c_search_steps: int. binary search steps for "confidence".
            c_range: turple. giving box constrains upper lower bounds.
            c_accuracy: float. minimum skip distance for "confidence" search.
            k_threshold: float. The higher k_threshold is, the higher f6 target loss must meet.
            verbose: bool. log attack process if true.

        Returns:
            Adversary instance with possible changed status.
        """
        if not adversary.is_targeted_attack:
            raise ValueError("This attack method only support targeted attack!")

        norm = self.norm
        learning_rate = self.epsilon_stepsize
        epsilon_ball = self.epsilon_ball

        # one hot encoder
        target_class = adversary.target_label
        num_labels = self.model.num_classes()
        assert target_class < num_labels
        if adversary.is_targeted_attack:
            target_onehot = paddle.eye(num_labels)[target_class]

        box_constrains_lower_bound = self.sample_float_range[0]
        box_constrains_upper_bound = self.sample_float_range[1]

        original_img = adversary.denormalized_original
        original_img = np.clip(original_img, box_constrains_lower_bound, box_constrains_upper_bound)
        mid_point = (box_constrains_upper_bound + box_constrains_lower_bound) * 0.5
        half_range = (box_constrains_upper_bound - box_constrains_lower_bound) * 0.5
        original_img_tensor = paddle.to_tensor(original_img, dtype='float32', place=self._device)

        c_lower_bound = c_range[0]
        c_upper_bound = c_range[1]
        confidence = (c_lower_bound + c_upper_bound) / 2.0

        best_l2 = None
        best_pred_label = None
        best_perturb = None

        # Outer loop for linearly searching for c
        for outer_step in range(c_search_steps):
            # renew the arctanh_w_tensor
            # hard code self.safe_num=0.999999 to make sure img to (0, 1) range
            linear_to_01 = (original_img - mid_point) / half_range * self.safe_num
            arctanh_w = np.arctanh(linear_to_01)
            arctanh_w_tensor = paddle.to_tensor(arctanh_w, dtype='float32', place=self._device, stop_gradient=False)
            optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=[arctanh_w_tensor])

            if norm == 'L2':
                result_turple = self._cwbL2(arctanh_w_tensor,
                                            original_img_tensor,
                                            target_onehot,
                                            confidence,
                                            k_threshold,
                                            attack_iterations,
                                            half_range,
                                            mid_point,
                                            optimizer,
                                            epsilon_ball,
                                            verbose=verbose)
                small_l2, current_pred_label, small_perturbed, small_perturbed_normalized = result_turple
            elif norm == 'Linf':
                # TODO: add CW, Linf attack
                print('developing')
            else:
                exit(1)

            if small_l2 is not None:
                c_upper_bound = confidence
                if best_l2 is None or small_l2 < best_l2:
                    best_l2 = small_l2
                    best_pred_label = current_pred_label
                    best_perturb = small_perturbed
                    best_perturb_normalized = small_perturbed_normalized
            else:
                c_lower_bound = confidence

            confidence_new = (c_lower_bound + c_upper_bound) / 2.0

            # end of reaching maximum accuracy
            if abs(confidence_new - confidence) <= c_accuracy:
                break
            if verbose:
                print("outer_step={} confidence {}->{}".format(outer_step, confidence, confidence_new))
            confidence = confidence_new

            if best_perturb is not None:
                best_perturb = np.squeeze(best_perturb)
                best_perturb_normalized = np.squeeze(best_perturb_normalized)
                adversary.try_accept_the_example(best_perturb,
                                                 best_perturb_normalized,
                                                 best_pred_label)
            else:
                pass

        return adversary

    def _cwbL2(self,
               arctanh_w_tensor,
               original_img_tensor,
               target_onehot,
               confidence,
               k_threshold,
               attack_iterations,
               half_range,
               mid_point,
               optimizer,
               epsilon_ball,
               verbose=False):
        """
        Launch an attack process with a given CW confidence.
        Args:
            arctanh_w_tensor: paddle.Tensor. hidden variable for img.
            original_img_tensor: paddle.Tensor. img(numpy.ndarray) to paddle.Tensor
            target_onehot: paddle.Tensor. target to paddle.Tensor one hot.
            confidence: int. CW confidence for f6.
            k_threshold: int. target logits threshold for single optimization ends.
            attack_iterations: int. optimization step using adam for perturbation.
            half_range: float. to compute or recover between hidden variable and img.
            mid_point: float. to compute or recover between hidden variable and img.
            optimizer: paddle.Optimizer. adam optimizer for CW.
            epsilon_ball: float. Perturbation bounds.
            verbose: bool. log attack process if true.

        Returns:
            small_l2: numpy.ndarray. retuen None if find no AE.
            current_pred_label: numpy.ndarray.
            small_perturbed: numpy.ndarray.
        """
        small_l2 = None
        current_pred_label = None
        small_perturbed = None
        small_perturbed_normalized = None

        for iteration in range(attack_iterations):
            optimizer.clear_grad()
            perturbed_image = paddle.tanh(arctanh_w_tensor) * half_range + mid_point
            perturbed_image_normalized = self.normalize(paddle.squeeze(perturbed_image))
            if len(perturbed_image_normalized.shape) < 4:
                perturbed_image_normalized = paddle.unsqueeze(perturbed_image_normalized, axis=0)

            logits = self.model.predict_tensor(perturbed_image_normalized)
            eta = paddle.clip(perturbed_image - original_img_tensor, -epsilon_ball, epsilon_ball)
            perturbed_image = original_img_tensor + eta
            l2_loss = paddle.sum((perturbed_image - original_img_tensor) ** 2)

            f6 = paddle.max(logits * (1 - target_onehot)) - paddle.max(logits * target_onehot)
            f6_loss = paddle.clip(f6, min=-k_threshold)

            l1 = confidence * f6_loss
            loss = l1 + l2_loss

            loss.backward(retain_graph=True)
            optimizer.step()

            l2_loss_np = l2_loss.numpy()
            logits_np = logits.numpy()
            perturbed_image_np = perturbed_image.numpy()
            perturbed_image_normalized_np = perturbed_image_normalized.numpy()

            if np.argmax(logits_np) == np.argmax(target_onehot):
                if small_l2 is None or l2_loss_np < small_l2:
                    small_l2 = l2_loss_np
                    current_pred_label = np.argmax(logits_np)
                    small_perturbed = perturbed_image_np
                    small_perturbed_normalized = perturbed_image_normalized_np

            if verbose:
                print("iteration={}, target label={}, "
                      "logits label={}, loss={}, l1={}, l2={}".format(iteration,
                                                                      np.argmax(target_onehot),
                                                                      np.argmax(logits.numpy()),
                                                                      loss.numpy(),
                                                                      l1.numpy(),
                                                                      l2_loss.numpy()))
        return small_l2, current_pred_label, small_perturbed, small_perturbed_normalized


CW_L2 = CWL2Attack
