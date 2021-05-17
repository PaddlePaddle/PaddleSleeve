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

import sys
sys.path.append("../../")

import abc
import logging

import paddle

from typing import List
from paddle import Tensor
from metrics.metrics import *
import metrics as met

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

    def evaluate(self, target, result, metrics, **kwargs) -> List[Tensor]:
        """
        Evaluate target and result using metrics

        Args:
            target(Tensor): Attack target (expected result)
            result(Tensor): Attack result (real result)
            metrics(Tensor): Metric names for evaluating

        Returns:
            (Tensor): Evaluate result
        """
        ret = []
        for metric in metrics:
            if metric not in met.metrics.__all__:
                raise ValueError("input metric {} is not in supported metrics set {}"
                    .format(metric, met.metrics.__all__))
            ret.append(globals()[metric](target, result))
        return ret

class InversionAttack(Attack):
    """ 
    Abstract model inversion attack class
    """

    @abc.abstractmethod
    def reconstruct(self, **kwargs) -> List[Tensor]:
        """
        Reconstruct target trained data from InversionAttack

        Returns:
            (Tensor): reconstructed data
        """
        raise NotImplementedError

class ExtractionAttack(Attack):
    """ 
    Abstract model extraction attack class
    """

    @abc.abstractmethod
    def extract(self, data, **kwargs) -> paddle.nn.Layer:
        """
        Extract target models

        Args:
            data(Tensor): input data that used for model extraction

        Returns:
            (Layer): extracted model
        """
        raise NotImplementedError

class InferenceAttack(Attack):
    """ 
    Abstract model inference attack class
    """

    @abc.abstractmethod
    def infer(self, data, **kwargs) -> paddle.Tensor:
        """
        Infer data's relationship with training set

        Args:
            data(Tensor): input data to infer its relationship (whether in training set)

        Returns:
            (Tensor): infer result
        """
        raise NotImplementedError

class MembershipInferenceAttack(InferenceAttack):
    """ 
    Abstract membership inference attack class
    """

    @abc.abstractmethod
    def infer(self, data, **kwargs) -> paddle.Tensor:
        """
        Infer whether data is in training set

        Args:
            data(Tensor): input data to infer its membership (whether in training set)

        Returns:
            (Tensor): infer result
        """
        raise NotImplementedError

class PropertyInferenceAttack(InferenceAttack):
    """ 
    Abstract property inference attack class
    """

    params = InferenceAttack.params + ["target_feature"]

    @abc.abstractmethod
    def infer(self, data, **kwargs) -> paddle.Tensor:
        """
        Infer properties from PropertyInferenceAttack

        Args:
            data(Tensor): input data that used to infer properties

        Returns:
            (Tensor): infer result
        """
        raise NotImplementedError