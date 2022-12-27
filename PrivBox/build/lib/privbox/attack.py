#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
Abstract base class for attacks
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import logging

import paddle

from typing import List
from paddle import Tensor
from .metrics import *

class Attack(abc.ABC):
    """
    Abstract base class for attacks
    """

    params: List[str] = list()

    def __init__(self):
        pass

    def set_params(self, **kwargs) -> None:
        """
        Set parameters for attacks

        Args:
            kwargs(dict): Parameters of dictionary type
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        # check params
        self.__check_params()

    def __check_params(self) -> None:
        pass

    def evaluate(self, target, result, metric_list, **kwargs) -> List[float]:
        """
        Evaluate target and result using metrics

        Args:
            target: Attack target (expected result)
            result: Attack result (real result)
            metrics(List[Metric]): Metric list

        Returns:
            (List[float]): Evaluate result
        """
        for metric in metric_list:
            if not isinstance(metric, Metric):
                raise ValueError("input metrics type error.")
        ret = []
        for metric in metric_list:
            ret.append(metric.compute(result, target))
        return ret
