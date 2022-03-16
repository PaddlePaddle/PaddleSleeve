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

"""Base class for metrics."""

from abc import ABC
from abc import abstractmethod
from functools import wraps
import warnings
import logging

from obj_detection.attack.utils.distances import MSE
from obj_detection.attack.utils.adversarial.detection import DetAdversarial
from obj_detection.attack.utils.adversarial.base import StopAttack


class Metric(ABC):
    """
    Abstract base class for DNN robustness metrics.

    The :class:`Metric` class represents a robustness testing metric
    that searches for adversarial examples with minimum perturbation.
    It should be subclassed when implementing new metrics.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be tested by the metric.
    criterion : a :class:`Criterion` instance
        The criterion that determines which images are adversarial.
    distance : a :class:`Distance` class
        The measure used to quantify similarity between images.
    threshold : float or :class:`Distance`
        If not None, the testing will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the test will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the test; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.

    Notes
    -----
    If a subclass overwrites the constructor, it should call the super
    constructor with args and kwargs.

    """

    def __init__(self,
                 model=None, criterion=None,
                 distance=MSE, threshold=None, verbose=True):
        self._default_model = model
        self._default_criterion = criterion
        self._default_distance = distance
        self._default_threshold = threshold
        self._verbose = verbose

        # to customize the initialization in subclasses, please
        # try to overwrite _initialize instead of __init__ if
        # possible
        self._initialize()

    def _initialize(self):
        """Additional initializer that can be overwritten by
        subclasses without redefining the full `__init__` method
        including all arguments and documentation.
        """
        pass

    @abstractmethod
    def __call__(self, input, **kwargs):
        raise NotImplementedError

    def name(self):
        """Returns a human readable name that uniquely identifies
        the metric with its hyperparameters.

        Returns
        -------
        str
            Human readable name that uniquely identifies the metric
            with its hyperparameters.

        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.

        """
        return self.__class__.__name__


def call_decorator(call_fn):
    """Decorator.
    """
    @wraps(call_fn)
    def wrapper(self, input, original_pred=None, unpack=True, threshold=None, **kwargs):
        """wrapper.
        """
        assert input is not None
        a = input
        """This part of the code initializes the adversary."""

        if input is None:
            raise ValueError('original image must be passed')
        else:
            model = self._default_model
            criterion = self._default_criterion
            distance = self._default_distance
            limit = threshold if threshold is not None else self._default_threshold

            if model is None or criterion is None:
                raise ValueError('The attack needs to be initialized '
                                 'with a model and a criterion.')

            a = DetAdversarial(
                    model, criterion, input, original_pred,
                    distance=distance, threshold=limit, verbose=self._verbose)

        assert a is not None

        """This part of the code runs the metric."""
        if a.distance.value == 0.:
            warnings.warn('Not running the attack because the original input'
                          ' is already misclassfied.')

        elif a.reached_threshold():
            warnings.warn('Not running the attack because the given threshold'
                          ' is already reached.')

        else:

            try:
                _ = call_fn(self, a, unpack=None, annotation=None, **kwargs)
                assert _ is None, 'decorated __call__ method must return None'

            except StopAttack:
                logging.info('threshold reached, stopping attack')

        if a.image is None:
            warnings.warn('{} did not find an adversarial, maybe the model'
                          ' or the criterion is not supported by this'
                          ' attack.'.format(self.name()))
        if unpack:
            return a.image
        else:
            return a

    return wrapper
