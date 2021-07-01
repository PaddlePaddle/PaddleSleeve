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

"""Classification model wrapper for Keras."""

from __future__ import absolute_import
import numpy as np
import logging
from perceptron.models.base import DifferentiableModel


class KerasModel(DifferentiableModel):
    """Create a :class:`Model` instance from a `Keras` model.

    Parameters
    ----------
    model : `keras.model.Model`
        The `Keras` model that are loaded.
    bounds  : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.
    """

    def __init__(
            self,
            model,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(KerasModel, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis,
                                         preprocessing=preprocessing)
        from keras import backend as K
        import keras

        self._task = 'cls'

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        images_input = model.input
        label_input = K.placeholder(shape=(1,))

        predictions = model.output

        shape = K.int_shape(predictions)
        _, num_classes = shape
        assert num_classes is not None
        self._num_classes = num_classes

        if predicts == 'probabilities':
            if K.backend() == 'tensorflow':
                predictions, = predictions.op.inputs
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=True)
        elif predicts == 'logits':
            loss = K.sparse_categorical_crossentropy(
                label_input, predictions, from_logits=True)

        # sparse_categorical_crossentropy returns 1-dim tensor,
        # gradients wants 0-dim tensor

        loss = K.squeeze(loss, axis=0)
        grads = K.gradients(loss, images_input)

        grad_loss_output = K.placeholder(shape=(num_classes, 1))
        external_loss = K.dot(predictions, grad_loss_output)
        # remove batch dimension of predictions
        external_loss = K.squeeze(external_loss, axis=0)
        # remove singleton dimension of grad_loss_output
        external_loss = K.squeeze(external_loss, axis=0)

        grads_loss_input = K.gradients(external_loss, images_input)

        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar tensor
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]

        self._loss_fn = K.function(
            [images_input, label_input], [loss])
        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        self._pred_grad_fn = K.function(
            [images_input, label_input], [predictions, grad])
        self._bw_grad_fn = K.function(
            [grad_loss_output, images_input], [grad_loss_input])

    def _to_logits(self, predictions):
        from keras import backend as K
        eps = 10e-8
        predictions = K.clip(predictions, eps, 1 - eps)
        predictions = K.log(predictions)
        return predictions

    def num_classes(self):
        """Return number of classes."""
        return self._num_classes

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task

    def batch_predictions(self, images):
        """Batch prediction of images."""
        px, _ = self._process_input(images)
        predictions = self._batch_pred_fn([px])
        assert len(predictions) == 1
        predictions = predictions[0]
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        """Returns both predictions and gradients."""
        input_shape = image.shape
        px, dpdx = self._process_input(image)
        predictions, gradient = self._pred_grad_fn([
            px[np.newaxis],
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == input_shape
        return predictions, gradient

    def backward(self, gradient, image):
        """Get gradients w.r.t. the original image."""
        assert gradient.ndim == 1
        gradient = np.reshape(gradient, (-1, 1))
        px, dpdx = self._process_input(image)
        gradient = self._bw_grad_fn([
            gradient,
            px[np.newaxis],
        ])
        gradient = gradient[0]  # output of bw_grad_fn is a list
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == image.shape
        return gradient
