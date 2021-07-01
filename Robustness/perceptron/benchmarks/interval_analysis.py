# Copyright 2021 Baidu Inc.
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

"""Interval analysis for evaluating model robustness."""

import warnings
import logging
import numpy as np
from tqdm import tqdm
from abc import ABC
from abc import abstractmethod
from .base import Metric
from .base import call_decorator
from perceptron.utils.image import onehot_like
from perceptron.utils.func import to_tanh_space
from perceptron.utils.func import to_model_space
from perceptron.utils.func import AdamOptimizer
from perceptron.utils.interval_analyze import symbolic, naive


class IntervalMetric(Metric, ABC):
    """The base class of interval analysis used for network 
    formal verifications.
    
    This verification method is described in [1,2]_. This 
    implementation is based on the symbolic interval lib
    in [3]_.

    References
    ----------
    .. [1] Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, 
        Suman Jana : "Formal Security Analysis of Neural Networks 
        using Symbolic Intervals", https://arxiv.org/abs/1804.10829
    .. [2] Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, 
        Suman Jana : "Efficient Formal Safety Analysis of Neural 
        Networks", https://arxiv.org/abs/1809.08098
    .. [3] https://github.com/tcwangshiqi-columbia/symbolic_interval
    """

    @call_decorator
    def __call__(self, adv, optimal_bound=False, epsilon=None,
                 parallel=False, unpack=False, annotation=None,
                 normalize=False, threshold=0.001):
        """ The Linf version of interval analysis. It will add two
        parameters into adversarial: (1) is_verified: whether the
        sample is verified to be safe under given epsilon;
        (2) opt: the optimal bound for verifying the property safe

        Parameters
        ----------
        adv : An class:`Adversarial` instance
            Keep all the information needed
        optimal_bound : Bool
            Whether we need to locate the minimal Linf bound that
            can be verified to be safe.
        epsilon : float
            The Linf epsilon range.
            If optimal_bound is False, it will serve as the testing epsilon.
            If optimal_bound is True, it will serve as the starting epsilon
            for searching the optimal one.
        parallel : Bool
            whether to parallelize the testing
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        annotation : int
            The reference label of the original input.
        normalize : Bool
            Whether the input is normalized. Usually, MNIST will not be 
            normalized while cifar10 will be normalized with 0.225.
        threshold : float
            The minimal threshold for the binary search
        """

        assert epsilon is not None, "Provide an epsilon for verification!"

        a = adv
        del adv
        del annotation

        if not optimal_bound:

            is_verified = self.analyze(a, epsilon, parallel=parallel)
            if is_verified:
                print("The model is proved to be safe with "
                      "given epsilon {0:.3f}".format(epsilon))
            else:
                print("Can not be verified with given "
                      "epsilon {0:.3f}".format(epsilon))

        if optimal_bound:
            opt = self.analyze_bound(a, epsilon, threshold=0.001, parallel=parallel)
            if normalize:
                opt = opt * 0.225 * 255
            else:
                opt = opt * 255
            print("optimal bound found to be {0:.3f} out of 0 to 255".format(opt))

        # Avoid warning for not finding the adversarial examples
        a._best_adversarial = a._original_image

        return None

    def analyze(self, adv, epsilon, parallel=False):
        """To be extended with different interval analysis methods."""
        raise NotImplementedError

    def analyze_bound(self, adv, epsilon, threshold=0.001, parallel=False):
        """ Return the optimal bound provided by interval analysis.
        It indicates the largest Linf bound that is verified to be 
        absent of adversarial examples under arbitrary attacks. The
        optimal bound is located by binary search.

        Parameters
        ----------
        adv : An class:`Adversarial` instance
            Keep all the information needed
        epsilon : float
            The Linf epsilon range. Serves as the starting epsilon
            for searching the optimal one.
        threshold : float
            The minimal threshold for the binary search
        parallel : Bool
            whether to parallelize the testing
        """

        bound = epsilon
        upper_bound = 1
        lower_bound = 0

        # binary search for the optimal bound
        while upper_bound - lower_bound > threshold:

            is_verified = self.analyze(adv, bound, parallel=parallel)

            if is_verified:
                # print("The model is proved to be safe with given epsilon", bound)
                lower_bound = bound
            else:
                # print("can not be verified with given epsilon", bound)
                upper_bound = bound

            bound = (upper_bound + lower_bound) / 2.0

        return bound


class SymbolicIntervalMetric(IntervalMetric):
    """ Symbolic interval Class """

    @staticmethod
    def analyze(adv, epsilon, parallel=False):
        """ Return whether the example is verified to be save
        within the given Linf <= epsilon analyzed by symbolic
        interval analysis.

        Parameters
        ----------
        adv : An class:`Adversarial` instance
            Keep all the information needed
        epsilon : float
            The Linf epsilon range for testing
        parallel : Bool
            whether to parallelize the testing
        """

        iloss, ierr = symbolic(adv._model._model, epsilon, \
                               adv._original_image, \
                               adv._original_pred, \
                               parallel=parallel)
        if ierr:
            is_verified = False
        else:
            is_verified = True

        return is_verified


class NaiveIntervalMetric(IntervalMetric):
    """ Naive interval analysis class """

    @staticmethod
    def analyze(a, epsilon, parallel=False):
        """ Return whether the example is verified to be save
        within the given Linf <= epsilon analyzed by naive
        interval analysis.

        Parameters
        ----------
        adv : An class:`Adversarial` instance
            Keep all the information needed.
        epsilon : float
            The Linf epsilon range for testing.
        parallel : Bool
            whether to parallelize the testing
        """
        iloss, ierr = naive(a._model._model, epsilon, \
                            a._original_image,
                            a._original_pred, \
                            parallel=parallel)
        if ierr:
            is_verified = False
        else:
            is_verified = True

        return is_verified
