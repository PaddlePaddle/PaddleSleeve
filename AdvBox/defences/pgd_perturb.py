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
PGD batch attack
"""
import os
import sys

# import cv2

# sys.path.append("../..")
import paddle
import paddle.nn.functional as F
import numpy as np
import copy

EPS = 1e-10


def denormalize_image(img, mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]):
    if len(img.shape) > 3:
        img = paddle.squeeze(img)
    for i in range(img.shape[0]):
        img[i] = img[i] * std[i] + mean[i]
    denorm = paddle.clip(img, min=0, max=1)
    return denorm


def pgd(model, img, label, norm='l2', epsilons=16/255, eps_step=2/255, steps=10,
        num_classes=10, model_mean=[0.5,0.5,0.5], model_std=[0.5,0.5,0.5]):
    """
    PGD batch attack on paddle2 image classification models
    Args:
        model (paddle.nn.layer): victim model
        img (3DTensor): normalized input image in format CHW
        label (int): class label
        norm: norm type, 'l2' or 'linf'
        epsilons: radius of eps ball
        eps_step: step size
        steps: num steps
        num_classes:
        model_mean:
        model_std:

    Returns:
    adv_norm (3DTensor): normalized adversarial example in format CHW
    """
    if label is None:
        raise ValueError("Attack method needs original label")
    norm = norm.lower()
    assert norm == 'l2' or norm == 'linf', 'Only support L2 or Linf norm'

    norm_fn = paddle.vision.transforms.Normalize(mean=model_mean, std=model_std)
    label_onehot = paddle.nn.functional.one_hot(label, num_classes=num_classes)

    _min, _max = (0, 1)
    C, H, W = img.shape

    ori_img_tensor = denormalize_image(img.clone(), mean=model_mean, std=model_std)
    adv_img_tensor = denormalize_image(img.clone(), mean=model_mean, std=model_std)

    model.eval()

    for i in range(steps):
        adv_img_tensor.stop_gradient = False
        adv_norm = norm_fn(adv_img_tensor)
        pred = model(paddle.unsqueeze(adv_norm, axis=0))
        cls = paddle.argmax(pred)

        # Test if the current adv succeed
        is_adv = cls != label
        if is_adv:
            return adv_norm

        loss = paddle.max(pred * label_onehot) - paddle.max(pred * (1 - label_onehot))
        loss.backward()
        grad_tensor = - adv_img_tensor.grad
        adv_img_tensor.stop_gradient = True

        if norm == 'l2':
            count_pix = np.sqrt(C * H * W * (_max - _min) ** 2)
            normalized_grad = grad_tensor / (paddle.norm(grad_tensor, p=2) + EPS) * count_pix
            adv_img_tensor = adv_img_tensor + normalized_grad * eps_step
            eta = adv_img_tensor - ori_img_tensor
            mse_eta = paddle.norm(eta, p=2) / count_pix
            if mse_eta > epsilons:
                eta *= epsilons / mse_eta
            adv_img_tensor = paddle.clip(ori_img_tensor + eta, _min, _max)
        else:
            normalized_grad = paddle.sign(grad_tensor)
            adv_img_tensor = adv_img_tensor + normalized_grad * eps_step
            eta = paddle.clip(adv_img_tensor - ori_img_tensor, -epsilons, epsilons)
            adv_img_tensor = paddle.clip(ori_img_tensor + eta, _min, _max)

    return norm_fn(adv_img_tensor).numpy()


class PGDTransform(object):
    """
    Base class for adversarial examples generation based on Paddle2
    model and Paddle adversarial attacks. Subclass should implement the
    _generate_adv_example(self, model_list, x, y) method.
    """
    def __init__(self, model, config_list, p=1):
        """
        Args:
            model: A paddle2 model to be attacked.
            config_list: A list of attack config corresponding to attack_methods.
            p (int): Attacking probability
        """
        np.random.seed(0)
        self._model = model
        self._attack_probability = p
        self._config_list = config_list

    def __call__(self, x_batch, y_batch):
        """
        Transform x to adv_x.
        Args:
            x_batch: A list of numpy.ndarray input samples.
            y_batch: A list of int or numpy.int input labels.
        Returns:
            Transformed adversarial examples
        """
        #x_batch = paddle.unstack(x_batch)
        adv_x_batch = []
        adv_y_batch = []
        for x, y in zip(x_batch, y_batch):
            if np.random.random(1) > self._attack_probability:
                adv_x_batch.append(np.squeeze(x))
                adv_y_batch.append(y)
            else:
                adv_x = pgd(self._model, x, y, **self._config_list)
                adv_x_batch.append(np.squeeze(adv_x))
                adv_y_batch.append(y)
        adv_x_batch = np.stack(adv_x_batch)
        adv_y_batch = np.stack(adv_y_batch)
        return adv_x_batch, adv_y_batch
