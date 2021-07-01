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

"""Provides a class that represents an adversarial example."""

import numpy as np
import numbers
from abc import ABC
from perceptron.utils.distances import MSE
from perceptron.utils.distances import Distance


class StopAttack(Exception):
    """Exception thrown to request early stopping of an attack
    if a given (optional!) threshold is reached.
    """
    pass


class Adversarial(ABC):
    """Defines the base class of an adversarial that should be found and
    stores the result. The :class:`Adversarial` class represents a single
    adversarial example for a given model, criterion and reference image.
    It can be passed to an adversarial attack to find the actual adversarial.
    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be evaluated against the adversarial.
    criterion : a :class:`Criterion` instance
        The criterion that determines which images are adversarial.
    original_image : a :class:`numpy.ndarray`
        The original image to which the adversarial image should
        be as close as possible.
    original_pred : int(ClsAdversarial) or dict(DetAdversarial)
        The ground-truth predictions of the original image.
    distance : a :class:`Distance` class
        The measure used to quantify similarity between images.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.
    """

    def __init__(
            self,
            model,
            criterion,
            original_image,
            original_pred=None,
            threshold=None,
            distance=MSE,
            verbose=False):

        self._model = model
        self._criterion = criterion
        self._original_image = original_image
        self._original_image_for_distance = original_image
        self._original_pred = original_pred
        self._distance = distance

        if threshold is not None and not isinstance(threshold, Distance):
            threshold = distance(value=threshold)
        self._threshold = threshold
        self.verbose = verbose
        self._best_adversarial = None
        self._best_distance = distance(value=np.inf)
        self._best_adversarial_output = None

        self._total_prediction_calls = 0
        self._total_gradient_calls = 0

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        # used for attacks that can provide a verifiable bound
        self._verifiable_bounds = (0., 0.)

        # check if the original image is already adversarial
        try:
            self.predictions(original_image)
        except StopAttack:
            # if a threshold is specified and the original input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self._distance.value == 0.

    def _reset(self):
        self._best_adversarial = None
        self._best_distance = self._distance(value=np.inf)
        self._best_adversarial_output = None

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        self.predictions(self._original_image)

    @property
    def verifiable_bounds(self):
        """The verifiable bounds obtained so far."""
        return self._verifiable_bounds

    @verifiable_bounds.setter
    def verifiable_bounds(self, bounds):
        """The setter of verifiable bounds"""
        self._verifiable_bounds = bounds

    @property
    def image(self):
        """The best adversarial found so far."""
        return self._best_adversarial

    @property
    def output(self):
        """The model predictions for the best adversarial found so far.
        None if no adversarial has been found.
        """
        return self._best_adversarial_output

    @property
    def distance(self):
        """The distance of the adversarial input to the original input."""
        return self._best_distance

    @property
    def original_image(self):
        """The original input."""
        return self._original_image

    @property
    def original_pred(self):
        """The original label."""
        return self._original_pred

    def set_distance_dtype(self, dtype):
        """Set the dtype of Distance."""
        assert dtype >= self._original_image.dtype
        self._original_image_for_distance = self._original_image.astype(
            dtype, copy=False)

    def reset_distance_dtype(self):
        """Reset the dtype of Distance."""
        self._original_image_for_distance = self._original_image

    def normalized_distance(self, image):
        """Calculates the distance of a given image to the
        original image.
        Parameters
        ----------
        image : `numpy.ndarray`
            The image that should be compared to the original image.
        Returns
        -------
        :class:`Distance`
            The distance between the given image and the original image.
        """
        return self._distance(
            self._original_image_for_distance,
            image,
            bounds=self.bounds())

    def reached_threshold(self):
        """Returns True if a threshold is given and the currently
        best adversarial distance is smaller than the threshold."""
        return self._threshold is not None \
               and self._best_distance <= self._threshold

    def target_class(self):
        """Interface to criterion.target_class for attacks.
        """
        try:
            target_class = self._criterion.target_class()
        except AttributeError:
            target_class = None
        return target_class

    def num_classes(self):
        """Return number of classes."""
        n = self._model.num_classes()
        assert isinstance(n, numbers.Number)
        return n

    def bounds(self):
        """Return bounds of model."""
        min_, max_ = self._model.bounds()
        assert isinstance(min_, numbers.Number)
        assert isinstance(max_, numbers.Number)
        assert min_ < max_
        return min_, max_

    def in_bounds(self, input_):
        """Check if input is in bounds."""
        min_, max_ = self.bounds()
        return min_ <= input_.min() and input_.max() <= max_

    def channel_axis(self, batch):
        """ Interface to model.channel_axis for attacks.
        Parameters
        ----------
        batch : bool
            Controls whether the index of the axis for a batch of images
            (4 dimensions) or a single image (3 dimensions) should be
            returned.
        """
        axis = self._model.channel_axis()
        if not batch:
            axis = axis - 1
            return axis

    def has_gradient(self):
        """ Returns true if _backward and _forward_backward can be called
        by an attack, False otherwise.
        """
        try:
            self._model.gradient
            self._model.predictions_and_gradient
        except AttributeError:
            return False
        else:
            return True

    def _new_adversarial(self, image, predictions, in_bounds):
        image = image.copy()  # to prevent accidental inplace changes
        distance = self.normalized_distance(image)
        if in_bounds and self._best_distance > distance:
            # new best adversarial
            if self.verbose:
                print('new best adversarial: {}'.format(distance))

            self._best_adversarial = image
            self._best_distance = distance
            self._best_adversarial_output = predictions

            self._best_prediction_calls = self._total_prediction_calls
            self._best_gradient_calls = self._total_gradient_calls

            if self.reached_threshold():
                raise StopAttack

            return True, distance
        return False, distance

    def _is_adversarial(self, image, predictions, in_bounds):
        """Interface to `criterion.is_adversary()` that calls
        _new_adversarial if necessary.
        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
        predictions : :class:`numpy.ndarray`
            A vector with the predictions for some image.
        label : int
            The label of the unperturbed reference image.
        """
        is_adversarial = self._criterion.is_adversarial(
            predictions, self._original_pred)
        assert isinstance(is_adversarial, bool) or \
               isinstance(is_adversarial, np.bool_)
        if is_adversarial:
            is_best, distance = self._new_adversarial(
                image, predictions, in_bounds)
        else:
            is_best = False
            distance = None
        return is_adversarial, is_best, distance

    def predictions(self, image, strict=True, return_details=False):
        """Interface to model.predictions for attacks.
        Parameters
        ----------
        image : `numpy.ndarray`
            Image with shape (height, width, channels).
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        in_bounds = self.in_bounds(image)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        predictions = self._model.predictions(image)
        is_adversarial, is_best, distance = self._is_adversarial(
            image, predictions, in_bounds)

        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def batch_predictions(
            self, images, greedy=False, strict=True, return_details=False):
        """Interface to model.batch_predictions for attacks.
        Parameters
        ----------
        images : `numpy.ndarray`
            Batch of images with shape (batch, height, width, channels).
        greedy : bool
            Whether the first adversarial should be returned.
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        if strict:
            in_bounds = self.in_bounds(images)
            assert in_bounds

        self._total_prediction_calls += len(images)
        predictions = self._model.batch_predictions(images)

        assert predictions.ndim == 2
        assert predictions.shape[0] == images.shape[0]

        if return_details:
            assert greedy

        adversarials = []
        for i in range(len(predictions)):
            if strict:
                in_bounds_i = True
            else:
                in_bounds_i = self.in_bounds(images[i])
            is_adversarial, is_best, distance = self._is_adversarial(
                images[i], predictions[i], in_bounds_i)
            if is_adversarial and greedy:
                if return_details:
                    return predictions, is_adversarial, i, is_best, distance
                else:
                    return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if greedy:  # pragma: no cover
            # no adversarial found
            if return_details:
                return predictions, False, None, False, None
            else:
                return predictions, False, None

        is_adversarial = np.array(adversarials)
        assert is_adversarial.ndim == 1
        assert is_adversarial.shape[0] == images.shape[0]

        return predictions, is_adversarial

    def gradient(self, image=None, label=None, strict=True):
        """Interface to model.gradient for attacks.
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
        raise NotImplementedError

    def predictions_and_gradient(
            self, image=None, label=None, strict=True, return_details=False):
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
        raise NotImplementedError

    def backward(self, gradient, image=None, strict=True):
        """ backward """
        raise NotImplementedError
