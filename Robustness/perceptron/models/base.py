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

""" Base class for models. """

from abc import ABC
from abc import abstractmethod
import numpy as np


def _create_preprocessing_fn(params):
    mean, std = params
    mean = np.asarray(mean)
    std = np.asarray(std)

    def identity(x):
        return x

    if np.all(mean == 0) and np.all(std == 1):
        def preprocessing(x):
            return x, identity

    elif np.all(std == 1):
        def preprocessing(x):
            _mean = mean.astype(x.dtype)
            return x - _mean, identity

    elif np.all(mean == 0):
        def preprocessing(x):
            _std = std.astype(x.dtype)

            def grad(dmdp):
                return dmdp / _std

            return x / _std, grad

    else:
        def preprocessing(x):
            _mean = mean.astype(x.dtype)
            _std = std.astype(x.dtype)
            result = x - _mean
            result /= _std

            def grad(dmdp):
                return dmdp / _std

            return result, grad

    return preprocessing


class Model(ABC):
    """Base class to provide metrics with a unified interface to models."""

    def __init__(self, bounds, channel_axis, preprocessing=(0, 1)):
        assert len(bounds) == 2
        self._bounds = bounds
        self._channel_axis = channel_axis
        if not callable(preprocessing):
            preprocessing = _create_preprocessing_fn(preprocessing)
        assert callable(preprocessing)
        self._preprocessing = preprocessing

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def bounds(self):
        return self._bounds

    def channel_axis(self):
        return self._channel_axis

    def _process_input(self, x):
        p, grad = self._preprocessing(x)
        if hasattr(p, 'dtype'):
            assert p.dtype == x.dtype
        p = np.asarray(p, dtype=x.dtype)
        assert callable(grad)
        return p, grad

    def _process_gradient(self, backward, dmdp):
        """
        backward: `callable`
            callable that backpropagates the gradient of the model w.r.t to
            preprocessed input through the preprocessing to get the gradient
            of the model's output w.r.t. the input before preprocessing
        dmdp: gradient of model w.r.t. preprocessed input
        """

        if backward is None:
            raise ValueError('Your preprocessing function does not provide'
                             ' an (approximate) gradient')

        dmdx = backward(dmdp)
        assert dmdx.dtype == dmdp.dtype
        return dmdx

    @abstractmethod
    def predictions(self, image):
        """Calculate prediction for a single image."""
        raise NotImplementedError


class DifferentiableModel(Model):
    """ Base class for differentiable models that provide gradients.
    The :class:`DifferentiableModel` class can be used as a base class
    for models that provide gradients. Subclasses must implement
    :meth:`predictions_and_gradient`.
    """

    @abstractmethod
    def batch_predictions(self, images):
        """Calculates predictions for a batch of images.
        Parameters
        ----------
        images : `numpy.ndarray`
            Batch of images with shape (batch size, height, width, channels).
        Returns
        -------
        `numpy.ndarray`
            Predictions (logits, or with bounding boxes).
        """
        raise NotImplementedError

    def predictions(self, image):
        """Convenience method that calculates predictions for a single image."""
        predictions = self.batch_predictions(np.expand_dims(image, 0))

        if isinstance(predictions, list):
            # Object detection models will return a list of preds in batch
            if len(predictions) > 0:
                return predictions[0]
            else:
                return None
        else:
            # Classification models will return a 2-D ndarray
            return np.squeeze(predictions, axis=0)

    @abstractmethod
    def num_classes(self):
        """Determines the number of classes."""
        raise NotImplementedError

    @abstractmethod
    def predictions_and_gradient(self, image, label):
        """Calculates predictions for an image and the gradient of
        the cross-entropy loss w.r.t. the image.
        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
        label : int
            Reference label used to calculate the gradient.
        Returns
        -------
        predictions : `numpy.ndarray`
            Vector of predictions. (logits, or with bounding boxes).
        gradient : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the image.
            Will have the same shape as the image.
        """
        raise NotImplementedError

    def gradient(self, image, label):
        """ Calculates the gradient of the cross-entropy loss w.r.t. the image.
        The default implementation calls predictions_and_gradient.
        Subclasses can provide more efficient implementations that only
        calculate the gradient.
        Parameters
        ----------
        image : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the image. Will
            have the same shape as the image.
        """
        _, gradient = self.predictions_and_gradient(image, label)
        return gradient

    @abstractmethod
    def backward(self, gradient, image):
        """ Backpropagates the gradient of some loss w.r.t. the logits
        through the network and returns the gradient of that loss w.r.t.
        the input image.
        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        image : `numpy.ndarry`
            Image with shape (height, width, channels).
        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t. the image.
        """
        raise NotImplementedError
