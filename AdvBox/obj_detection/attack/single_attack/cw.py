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
import logging
import numpy as np
import paddle
from .base import Metric
from .base import call_decorator
from obj_detection.attack.utils.func import to_tanh_space
from obj_detection.attack.utils.distances import MSE, Linf

from PIL import Image

__all__ = ['CarliniWagnerMetric', 'CW']

from ..utils.tools import plot_image_objectdetection_ppdet


class CarliniWagnerMetric(Metric):
    """
    Uses Adam to minimize the CW L2 objective function
    Paper link: https://arxiv.org/abs/1608.04644
    Args:
        model: PaddleWhiteBoxModel.
        learning_rate: float. for adam optimizer.
    """

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 c_search_steps=5, max_iterations=10,
                 confidence=0, learning_rate=5e-2,
                 c_init=0.1, abort_early=False):
        """
        Launch a CW attack process.
        Args:
        adv: Adversary.
            An adversary instance with initial status.
        label : int.
            The reference label of the original input.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        c_search_steps : int
            The number of steps for the binary search used to find the
            optimal tradeoff-constant between distance and confidence.
        max_iterations : int
            The maximum number of iterations. Larger values are more
            accurate; setting it too small will require a large learning
            rate and will produce poor results.
        confidence : int or float
            Confidence of adversarial examples: a higher value produces
            adversarials that are further away, but more strongly classified
            as adversarial.
        learning_rate : float
            The learning rate for the attack algorithm. Smaller values
            produce better results but take longer to converge.
        c_init : float
            The initial tradeoff-constant to use to tune the relative
            importance of distance and confidence. If `binary_search_steps`
            is large, the initial constant is not important.
        abort_early : bool
            If True, attack will finish once an adversarial example is found

        Returns:
            Adversary instance with possible changed status.
        """
        self._device = paddle.get_device()

        if adv.target_class() == None:
            raise ValueError("This attack method only support targeted attack!")
        if isinstance(adv.distance, MSE):
            self.norm = 'l2'
        elif isinstance(adv.distance, Linf):
            self.norm = 'linf'
        else:
            raise ValueError("This attack method only support l2 or linf distance!")

        min_, max_ = (0, 1)
        mid_point = (min_ + max_) * 0.5
        half_range = (max_ - min_) * 0.5

        adv_img = np.copy(adv.original_image)
        # variables representing inputs in attack space will be
        # prefixed with tanh_
        tanh_original = to_tanh_space(adv_img, min_, max_)
        # will be close but not identical to a.original_image
        reconstructed_original = np.tanh(tanh_original) * half_range + mid_point
        reconstructed_original_tensor = paddle.to_tensor(reconstructed_original, dtype='float32',
                                                         place=self._device, stop_gradient=False)

        C, H, W = adv_img.shape
        self._target_class = adv.target_class()
        self._confidence = confidence

        # the binary search finds the smallest const for which we find an adversarial
        const = c_init
        c_lower_bound = 0
        c_upper_bound = 10
        best_lp = None
        best_perturbed = None
        best_pred = None
        adv._model._model.eval()

        # Outer loop for linearly searching for c
        for c_step in range(c_search_steps):
            if c_step == c_search_steps - 1 and c_search_steps >= 10:
                const = c_upper_bound

            logging.info('starting optimization with const = {}'.format(const))
            tanh_pert = np.copy(tanh_original)

            tanh_pert_tensor = paddle.to_tensor(tanh_pert, dtype='float32', place=self._device, stop_gradient=False)

            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf

            small_lp = None
            small_perturbed = None
            small_pred = None

            # create a new optimizer to minimize the perturbation
            optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=[tanh_pert_tensor])

            for iteration in range(max_iterations):
                optimizer.clear_grad()
                x = paddle.tanh(tanh_pert_tensor) * half_range + mid_point
                x_norm = adv._model._preprocessing(x)
                adv._model._model.eval()
                features = adv._model._gather_feats(x_norm)
                # Test if the current adv succeed
                
                is_adv, is_best, distance = adv._is_adversarial(np.squeeze(x.numpy()),
                                                                features['bbox_pred'].numpy(), True)
                if is_adv and abort_early:
                    return

                adv_loss = adv._model.adv_loss(features=features, target_class=self._target_class)

                if self.norm == 'l2':
                    lp_loss = paddle.sum((x - reconstructed_original_tensor) ** 2) / np.sqrt(C * H * W)
                else:
                    lp_loss = paddle.max(paddle.abs(x - reconstructed_original_tensor)).astype(np.float64)
                loss = lp_loss + const * adv_loss
                loss.backward(retain_graph=True)
                optimizer.step()

                lp_loss_np = lp_loss.numpy()
                if is_adv:
                    if small_lp is None or lp_loss_np < small_lp:
                        small_lp = lp_loss_np
                        small_perturbed = x.numpy()
                        small_pred = features['bbox_pred'].numpy()

            if small_perturbed is not None:
                c_upper_bound = const
                if best_lp is None or small_lp < best_lp:
                    best_lp = small_lp
                    best_perturbed = small_perturbed
                    best_pred = small_pred
            else:
                c_lower_bound = const

            const_new = (c_lower_bound + c_upper_bound) / 2.0

            # # end of reaching maximum accuracy
            # if abs(confidence_new - confidence) <= c_accuracy:
            #     break
            # if verbose:
            #     print("outer_step={} confidence {}->{}".format(outer_step, confidence, confidence_new))
            const = const_new

            if best_perturbed is not None:
                best_perturbed = np.squeeze(best_perturbed)
                best_pred = np.squeeze(best_pred)
#                 adv._best_adversarial = best_perturbed
#                 adv._best_distance = best_l2
#                 adv._best_adversarial_output = best_pred

            else:
                pass

    def _plot_adv_and_diff(self, adv, adv_img, best_pred, ori_img, ori_pred):
        adv_image = Image.fromarray((np.transpose(adv_img, [1,2,0]) * 255).astype(np.uint8))
        adv_image_clear = (np.transpose(adv_img, [1,2,0]) * 255).astype('uint8')
        ori_image = Image.fromarray((np.transpose(ori_img, [1,2,0]) * 255).astype('uint8'))

        adv_image = adv._model._draw_bbox(adv_image, best_pred, 0.3)
        ori_image = adv._model._draw_bbox(ori_image, ori_pred, 0.3)
#        adv_image_clear = cv2.resize(adv_image_clear, (640, 404), interpolation=2)
        adv_image_clear = Image.fromarray(adv_image_clear)
        adv_image_clear.save('cw_adv_clear.png', 'PNG')
        adv_image.save('cw_adv.png', "PNG")
        ori_image.save('cw_ori.png', "PNG")


CW = CarliniWagnerMetric
