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
This module provides the attack method of project gradient attack.
L2 distance metrics especially
"""
from __future__ import division
from __future__ import print_function

import copy
from builtins import range

import numpy as np
import paddle
from PIL import Image

from .base import Metric
from .base import call_decorator
from obj_detection.attack.utils.distances import MSE, Linf

__all__ = ['ProjectedGradientDescentMetric', 'PGD']

EPS = 1e-8

class ProjectedGradientDescentMetric(Metric):
    """
    Projected Gradient Descent
    Towards deep learning models resistant to adversarial attacks, A. Madry, A. Makelov, L. Schmidt, D. Tsipras,
    and A. Vladu, ICLR 2018
    """

    @call_decorator
    def __call__(self, adv, annotation=None, unpack=True,
                 abort_early=True, epsilons=20 / 255, eps_step=4 / 255,
                 steps=20, use_opt=False):
        """
        Launch a Projected Gradient Descent attack process.
        Args:
        adv: Adversary.
            An adversary instance with initial status.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        abort_early : bool
            If True, Adam will be aborted if the loss hasn't decreased for
            some time (a tenth of max_iterations).
        epsilons : float
            maximum amplitude allowed
        eps_step : float
            the step size we take at each iteration
        steps : int
            maximum numbers of iterations
        use_opt : bool
            use paddle.optimizer or calculate gradient directly

        Returns:
            Adversary instance with possible changed status.
        """
        self._device = paddle.get_device()

        if adv.target_class() == None:
            raise ValueError("This attack method only support targeted attack!")

        _min, _max = (0, 1)
        C, H, W = adv.original_image.shape

        self._target_class = adv.target_class()
        if isinstance(adv.distance, MSE):
            self.norm = 'L2'
        elif isinstance(adv.distance, Linf):
            self.norm = 'Linf'
        else:
            raise TypeError('Projected gradient descent attack only support MSE or Linf distance')

        ori_img = np.copy(adv.original_image)
        ori_img_tensor = paddle.to_tensor(ori_img, dtype='float32', place=self._device)
        adv_img_tensor = paddle.to_tensor(ori_img, dtype='float32', place=self._device, stop_gradient=False)
        if use_opt:
            opt = paddle.optimizer.SGD(learning_rate=eps_step, parameters=[adv_img_tensor])
        adv._model._model.eval()

        for i in range(steps):
            adv_img_tensor.stop_gradient=False
            x_norm = adv._model._preprocessing(adv_img_tensor)
            features = adv._model._gather_feats(x_norm)

            # Test if the current adv succeed
            is_adv, is_best, distance = adv._is_adversarial(np.squeeze(adv_img_tensor.numpy()),
                                                            features['bbox_pred'].numpy(),
                                                            True)
            if is_adv:
                adv_image = Image.fromarray((np.transpose(adv_img_tensor.numpy(), [1, 2, 0]) * 255).astype(np.uint8))
                return

            loss = adv._model.adv_loss(features=features, target_class=self._target_class)

            if use_opt:
                opt.clear_grad()
                loss.backward()
                opt.step()
            else:
                # loss.backward(retain_graph=True)
                # print(adv_img_tensor.gradient())
                loss.backward()
                grad_tensor = - adv_img_tensor.grad
                adv_img_tensor.stop_gradient=True

                if self.norm == 'L2':
                    count_pix = np.sqrt(C * H * W * (_max - _min) ** 2)
                    normalized_grad = grad_tensor / (paddle.norm(grad_tensor, p=2) + EPS) * count_pix
                    adv_img_tensor = adv_img_tensor + normalized_grad * eps_step
                    eta = adv_img_tensor - ori_img_tensor
                    mse_eta = paddle.norm(eta, p=2) / count_pix
                    if mse_eta > epsilons:
                        eta *= epsilons / mse_eta
                    adv_img_tensor = paddle.clip(ori_img_tensor + eta, _min, _max)
                elif self.norm == 'Linf':
                    normalized_grad = paddle.sign(grad_tensor)
                    adv_img_tensor = adv_img_tensor + normalized_grad * eps_step
                    eta = paddle.clip(adv_img_tensor - ori_img_tensor, -epsilons, epsilons)
                    adv_img_tensor = paddle.clip(ori_img_tensor + eta, _min, _max)
                else:
                    exit(-1)

    def _plot_adv_and_diff(self, adv, adv_img, best_pred, ori_img, ori_pred):
        adv_image = Image.fromarray((np.transpose(adv_img, [1, 2, 0]) * 255).astype(np.uint8))
        adv_image_clear = (np.transpose(adv_img, [1, 2, 0]) * 255).astype('uint8')
        ori_image = Image.fromarray((np.transpose(ori_img, [1, 2, 0]) * 255).astype('uint8'))

        adv_image = adv._model._draw_bbox(adv_image, best_pred, 0.3)
        ori_image = adv._model._draw_bbox(ori_image, ori_pred, 0.3)
        #        adv_image_clear = cv2.resize(adv_image_clear, (640, 404), interpolation=2)
        adv_image_clear = Image.fromarray(adv_image_clear)
        adv_image_clear.save('cw_adv_clear.png', 'PNG')
        adv_image.save('cw_adv.png', "PNG")
        ori_image.save('cw_ori.png', "PNG")


PGD = ProjectedGradientDescentMetric
