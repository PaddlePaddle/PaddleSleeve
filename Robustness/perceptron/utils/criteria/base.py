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
from abc import ABC
from abc import abstractmethod
import numpy as np


class Criterion(ABC):
    """Base class for criteria that define what is adversarial.
    The :class:`Criterion` class represents a criterion used to
    determine if predictions for an image are adversarial given
    a reference label. It shoud be subclassed when implementing
    new criteria. Subclasses must implement is_adversarial.
    """

    def name(self):
        """Returns a human readable name."""
        return self.__class__.__name__

    @abstractmethod
    def is_adversarial(self, predictions, ground_truth):
        """Decides if predictions for an image are adversarial given
        a reference ground truth.
        """

        raise NotImplementedError

    def __and__(self, other):
        return CombinedCriteria(self, other)


class CombinedCriteria(Criterion):
    """Meta criterion that combines several criteria into a new one.
    Parameters
    ----------
    *criteria : variable length list of :class:`Criterion` instances
        List of sub-criteria that will be combined.
    Notes
    -----
    This class uses lazy evaluation of the criteria in the order they are
    passed to the constructor.
    """

    def __init__(self, *criteria):
        super(CombinedCriteria, self).__init__()
        self._criteria = criteria

    def name(self):
        """ Concatenates the names of the given criteria in alphabetical
        order."""
        names = (criterion.name() for criterion in self._criteria)
        return '__'.join(sorted(names))

    def is_adversarial(self, predictions, ground_truth):
        for criterion in self._criteria:
            if not criterion.is_adversarial(predictions, ground_truth):
                return False

        return True
