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

"""Provide base classes that define what is adversarial."""

import sys
from perceptron.utils.func import softmax
from .base import Criterion
import numpy as np


class Misclassification(Criterion):
    """Defines adversarials as images for which the predicted class
    is not the original class.
    """

    def name(self):
        """Return criterion name."""
        return 'Top1Misclassification'

    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial."""
        top1 = np.argmax(predictions)
        return top1 != label


class ConfidentMisclassification(Criterion):
    """ Defines adversarials as images for which the probability
    of any class other than the original is above a given threshold.
    """

    def __init__(self, threshold):
        super(ConfidentMisclassification, self).__init__()
        assert 0 <= threshold <= 1
        self.threshold = threshold

    def name(self):
        """Return criterion name."""
        return '{}-{:.04f}'.format(self.__class__.__name__, self.threshold)

    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial."""
        top1 = np.argmax(predictions)
        probabilities = softmax(predictions)
        return (np.max(probabilities) >= self.threshold) and (top1 != label)


class TopKMisclassification(Criterion):
    """Defines adversarials as images for which the original class is
    not one of the top k predicted classes.
    For k=1, the :class:`Misclassification` class provides a more efficient
    implementation.
    Parameters
    ----------
    k : int
        Number of top predictions to which the reference label is compared to.
    """

    def __init__(self, k):
        super(TopKMisclassification, self).__init__()
        self.k = k

    def name(self):
        """Return criterion name."""
        return 'Top{}Misclassification'.format(self.k)

    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial."""
        topk = np.argsort(predictions)[-self.k:]
        return label not in topk


class TargetClass(Criterion):
    """Defines adversarials as images for which the predicted class is the given
    target class.
    Parameters
    ----------
    target_class : int
        The target class that needs to be predicted for an image
        to be considered an adversarial.
    """

    def __init__(self, target_class):
        super(TargetClass, self).__init__()
        self._target_class = target_class

    def target_class(self):
        """Return target class."""
        return self._target_class

    def name(self):
        """Return criterion name."""
        return '{}-{}'.format(self.__class__.__name__, self.target_class())

    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial."""
        top1 = np.argmax(predictions)
        return top1 == self.target_class()


class OriginalClassProbability(Criterion):
    """Defines adversarials as images for which the probability of original
    class is below a given threshold.
    This criterion alone does not guarantee that the class predicted for the
    adversarial image is not original class (unless p < 1 / num of classes).
    Therefore, it should usually be combined with a classification criterion.
    Parameters
    ----------
    p : float
        The threshold probability. If the probability of the original class is
        below this threshold, the image is considered an adversarial. It must
        satisfy 0 <= p <=1.
    """

    def __init__(self, p):
        super(OriginalClassProbability, self).__init__()
        assert 0 <= p <= 1
        self.p = p

    def name(self):
        """Return criterion name."""
        return '{}-{:.04f}'.format(self.__class__.__name__, self.p)

    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial."""
        probabilities = softmax(predictions)
        return probabilities[label] < self.p


class TargetClassProbability(Criterion):
    """Defines adversarials as images for which the probability
    of a given target class is above a given threshold.
    If the threshold is below 0.5, this criterion does not guarantee
    that the class predicted for the adversarial image is not the
    original class. In that case, it should usually be combined with
    a classification criterion.
    Parameters
    ----------
    target_class : int
        The target class for which the predicted probability must
        be above the threshold probability p, otherwise the image
        is not considered an adversarial.
    p : float
        The threshold probability. If the probability of the
        target class is above this threshold, the image is
        considered an adversarial. It must satisfy 0 <= p <= 1.
    """

    def __init__(self, target_class, p):
        super(TargetClassProbability, self).__init__()
        self._target_class = target_class
        assert 0 <= p <= 1
        self.p = p

    def target_class(self):
        """Return target class """
        return self._target_class

    def name(self):
        """Return criterion name."""
        return '{}-{}-{:.04f}'.format(
            self.__class__.__name__, self.target_class(), self.p)

    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial."""
        probabilities = softmax(predictions)
        return probabilities[self.target_class()] > self.p


class MisclassificationAntiPorn(Criterion):
    """Defines adversarials as image for which the probability of being
    `normal` is larger than the probability of being `porn`.
    """

    def name(self):
        """Return criterion name."""
        return 'AntiPornMisclassification'

    def is_adversarial(self, predictions, label):
        _porn_prob = 0
        _normal_prob = 0
        """Decides if predictions for an image are adversarial."""
        from perceptron.utils.tools import bcolors

        if 'error_code' in predictions:
            raise ValueError(bcolors.RED + 'API Error:' + str(predictions) + bcolors.ENDC)

        for cls in predictions:
            if cls['class_name'] == '色情':
                _porn_prob = cls['probability']
            if cls['class_name'] == '正常':
                _normal_prob = cls['probability']
        return _porn_prob < _normal_prob


class MisclassificationOfflineAntiPorn(Criterion):
    """Defines adversarials as image for which the probability of being
    `normal` is larger than the probability of being `porn`.
    """

    def name(self):
        """Return criterion name."""
        return 'AntiPornMisclassification'

    def is_adversarial(self, predictions, label):
        _porn_prob = 0
        _normal_prob = 0
        """Decides if predictions for an image are adversarial."""
        from perceptron.utils.tools import bcolors

        if predictions['err_no'] != 0:
            raise ValueError(bcolors.RED + 'API Error:' + predictions['err_msg'] + bcolors.ENDC)

        probabilities = predictions['result']['result'][0]['probability']
        _porn_prob = probabilities[0]
        _normal_prob = probabilities[2]
        return _porn_prob < _normal_prob


class MisclassificationSafeSearch(Criterion):
    """Defines adversarials as image for which the probability of being
    `unsafe` is lower than a threshold.
    Parameters
    ----------
    target_class : str
        The target `unsafe` class the you want to evade. It must
        be in one of ['adult', 'medical', 'racy', 'spoofed', 'violence'].
    threshold : int
        The threshold probability. If the probability of the
        target class is **below** this threshold, the image is
        considered an adversarial. It must satisfy 0 <= threshold <= 5.
    """

    def __init__(self, target_class='adult', threshold=2):
        super(MisclassificationSafeSearch, self).__init__()
        assert target_class in \
               ['adult', 'medical', 'racy', 'spoofed', 'violence']
        self._target_class = target_class
        assert isinstance(threshold, int) and 0 <= threshold <= 5
        self.threshold = threshold
        self.likelihood_name = {
            'UNKNOWN': 0,
            'VERY_UNLIKEYLY': 1,
            'UNLIKELY': 2,
            'POSSIBLE': 3,
            'LIKELY': 4,
            'VERY_LIKELY': 5
        }

    def target_class(self):
        """Return target class """
        return self._target_class

    def name(self):
        """Return criterion name."""
        return '{}-{}-{}'.format(
            self.__class__.__name__, self.target_class(), self.threshold)

    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial."""
        assert isinstance(predictions, dict), 'Predictions should be dict'
        probability = predictions[self._target_class]
        return probability <= self.threshold
