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

from __future__ import absolute_import

import numpy as np
import warnings
from perceptron.models.base import DifferentiableModel


class PyTorchModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `PyTorch` module.

    Parameters
    ----------
    model : `torch.nn.Module`
        The PyTorch model that are loaded.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    device : string
        A string specifying the device to do computation on.
        If None, will default to "cuda:0" if torch.cuda.is_available()
        or "cpu" if not.
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
            device=None,
            preprocessing=(0, 1)):

        # lazy import
        import torch

        super(PyTorchModel, self).__init__(bounds=bounds,
                                           channel_axis=channel_axis,
                                           preprocessing=preprocessing)

        self._num_classes = num_classes
        self._task = 'cls'

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self._model = model.to(self.device)

        if model.training:
            warnings.warn(
                'The PyTorch model is in training model and therefore'
                'might not be deterministic. Call the eval() method to'
                'set it in evaluation mode if this is not intended.')

    def batch_predictions(self, images):
        """Batch prediction of images."""
        # lazy import
        import torch

        images, _ = self._process_input(images)
        n = len(images)
        images = torch.from_numpy(images).to(self.device)

        predictions = self._model(images)
        predictions = predictions.to("cpu").detach()
        predictions = predictions.numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def num_classes(self):
        """Return number of classes."""
        return self._num_classes

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task

    def predictions_and_gradient(self, image, label):
        """Returns both predictions and gradients."""
        # lazy import
        import torch
        import torch.nn as nn
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target).long().to(self.device)

        images = image[np.newaxis]
        images = torch.from_numpy(images).to(self.device)
        images.requires_grad_()
        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = images.grad

        predictions = predictions.to("cpu").detatch().numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        grad = grad.to("cpu").detach().numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def _loss_fn(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn
        image, _ = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target).long().to(self.device)
        images = torch.from_numpy(image[None]).to(self.device)
        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss = loss.to("cpu")
        loss = loss.numpy()
        return loss

    def backward(self, gradient, image):
        """Get gradients w.r.t. the original image."""
        # lazy import
        import torch
        assert gradient.ndim == 1

        gradient = torch.from_numpy(gradient).to(self.device)

        input_shape = image.shape
        image, dpdx = self._process_input(image)
        images = image[np.newaxis]
        images = torch.from_numpy(images).to(self.device)
        images.requires_grad_()
        predictions = self._model(images)

        predictions = predictions[0]

        assert gradient.dim() == 1
        assert predictions.dim() == 1
        assert gradient.size() == predictions.size()

        loss = torch.dot(predictions, gradient)
        loss.backward()
        # should be the same as predictions.backward(gradient=gradient)

        grad = images.grad
        grad = grad.to("cpu").detach().numpy()
        grad = np.squeeze(grad, axis=0)
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return grad
