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

"""Classification model wrapper for Paddle."""

from __future__ import absolute_import

import numpy as np
import warnings
from perceptron.models.base import DifferentiableModel


class PaddleModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `Paddle` module.

    Parameters
    ----------
    model : `paddle.nn.Layer`
        The Paddle model that are loaded.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    """

    def __init__(
            self,
            model,
            bounds,
            num_classes,
            channel_axis=1,
            preprocessing=(0, 1)):
        """ Init function """
        # lazy import
        import paddle

        super(PaddleModel, self).__init__(bounds=bounds,
                                          channel_axis=channel_axis,
                                          preprocessing=preprocessing)
        self._num_classes = num_classes
        self._task = 'cls'
        self._model = model

        if model.training:
            warnings.warn(
                'The Paddle model is in training model and therefore'
                'might not be deterministic. Call the eval() method to'
                'set it in evaluation mode if this is not intended.')

    def batch_predictions(self, images):
        """Batch prediction of images."""
        # lazy import
        import paddle

        images, _ = self._process_input(images)
        n = len(images)
        images = paddle.to_tensor(images)

        predictions = self._model(images)
        predictions = predictions.detach()  # 从当前计算图分离
        predictions = predictions.numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def process_input_for_single(self, image):
        """Single image preprocessing."""
        images = np.expand_dims(image, 0)
        images, _ = self._process_input(images)
        return images[0]

    def num_classes(self):
        """Return number of classes."""
        return self._num_classes

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task

    def predictions_and_gradient(self, image, label):
        """Returns both predictions and gradients."""
        # lazy import
        import paddle
        import paddle.nn as nn
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        target = np.array([label])
        target = paddle.to_tensor(target)
        target = paddle.cast(target, 'int64')

        images = image[np.newaxis]  # 增加新的维度, [1, ..]
        images = paddle.to_tensor(images)
        images.stop_gradient = False  # 参与梯度计算

        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = images.grad  # 查看Tensor的梯度

        predictions = predictions.detach().numpy()
        predictions = np.squeeze(predictions, axis=0)  # batch_size 如果等于1，remove
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        grad = grad
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def _loss_fn(self, image, label):
        """ Loss function """
        # lazy import
        import paddle
        import paddle.nn as nn
        image, _ = self._process_input(image)
        target = np.array([label])
        target = paddle.to_tensor(target)
        target = paddle.cast(target, 'int64')

        images = image[None]  # 增加新的维度, [1, ..]
        images = paddle.to_tensor(images)

        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)

        loss = loss.numpy()
        return loss

    def backward(self, gradient, image):
        """Get gradients w.r.t. the original image."""
        # lazy import
        import paddle
        assert gradient.ndim == 1

        gradient = paddle.to_tensor(gradient)

        input_shape = image.shape
        image, dpdx = self._process_input(image)
        images = image[np.newaxis]
        images = paddle.to_tensor(images)
        images.stop_gradient = False  # 参与梯度计算

        predictions = self._model(images)[0]

        assert gradient.dim() == 1
        assert predictions.dim() == 1
        assert gradient.size == predictions.size

        loss = paddle.dot(predictions, gradient)  # 内积

        loss.backward()

        # should be the same as predictions.backward(gradient=gradient)

        grad = images.grad
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)

        assert grad.shape == input_shape

        return grad
