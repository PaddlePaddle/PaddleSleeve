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

"""Provides a class that represents an adversarial example
for object detection tasks.
"""

import numpy as np
import numbers
from .base import Adversarial
from .base import StopAttack
from perceptron.utils.distances import MSE
from perceptron.utils.distances import Distance


class DetAdversarial(Adversarial):
    """Defines an adversarial that should be found and stores the result."""

    def __init__(
            self,
            model,
            criterion,
            original_image,
            original_pred,
            threshold=None,
            distance=MSE,
            verbose=False):

        super(DetAdversarial, self).__init__(
            model,
            criterion,
            original_image,
            original_pred,
            threshold,
            distance,
            verbose)

        self._task = 'det'

    def model_task(self):
        """Interface to model.model_task for attacks."""
        return self._task

    def gradient(self, image=None, label=None, strict=True):
        """Interface to model.gradient for attacks.
        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
            Defaults to the original image.
        label : int
            Label used to calculate the loss that is differentiated.
            Deefaults to the original label
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        pass

    def predictions_and_gradient(
            self, image=None, annotation=None, strict=True, return_details=False):
        """Interface to model.predictions_and_gradient for attacks.
        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
            Defaults to the original image.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        assert self.has_gradient()

        if image is None:
            image = self._original_image

        assert not strict or self.in_bounds(image)

        in_bounds = self.in_bounds(image)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        self._total_gradient_calls += 1
        predictions, loss, gradient = \
            self._model.predictions_and_gradient(image, self._criterion)
        is_adversarial, is_best, distance = self._is_adversarial(
            image, predictions, in_bounds)

        assert gradient.shape == image.shape
        if return_details:
            return predictions, loss, gradient, is_adversarial, is_best, distance
        else:
            return predictions, loss, gradient, is_adversarial

    def backward(self, target_class, image=None, strict=True):
        """Interface to model.backward."""
        assert self.has_gradient()
        if image is None:
            image = self.__original_image

        assert not strict or self.in_bounds(image)
        self._total_gradient_calls += 1
        loss, gradient = self._model.backward(target_class, image)
        assert gradient.shape == image.shape
        return loss, gradient
