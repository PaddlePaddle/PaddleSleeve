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
A Finite-state Machine for AEs management.
It defines a class that contains the input object, the target and the
adversarial example.
"""
import numpy as np


class Denoising(object):
    """
    Usage philosophy:
    The objective of this tool box is to define a initial denosing instance and use
    denoising methods to transform it into a completed status (with a corresponding ADE).

    In completed status, the denoising.denoising_example() is not None.
    """
    def __init__(self, input, input_label, target_label=None):
        """
        Initialization for denoising object.
        Args:
            input: numpy.ndarray. The input sample, such as an image. Numpy
            input_label: numpy.ndarray. The input sample's label.
        """
        assert isinstance(input, np.ndarray)
        assert isinstance(target_label, int) or isinstance(target_label, np.int64)
        self.__input = input
        self.__input_label = input_label

        self.__target_label = target_label
        self.denoising_label = None

        self.__denoising_sample = None
        self.__bad_denoising_example = None

    def summary(self):
        """
        Print the summay of the denoising information
        Args:
            None.
        Returns:
            None.
        """

        print("input label:", self.__input_label)
        print("target label:", self.__target_label)
        print("denoising label:", self.denoising_label)
        print("contains a successful DE:", self.is_successful())

    def set_status(self, target_label=None):
        """
        Set the denoising instance to be:
        * targeted status.
        Args:
            target_label: int. This label should be the input label.
        Returns:
            None.
        """
        assert isinstance(target_label, int) or isinstance(target_label, np.int64)
        assert target_label is not None
        self.__target_label = target_label

    def reset(self):
        """
        Reset denoising status.
        Returns:
            None.
        """
        self.__target_label = None
        self.denoising_label = None

        self.__denoising_sample = None
        self.__bad_denoising_example = None

    def _is_successful(self, denoising_label):
        """
        Check if the denoising_label is the expected true label.
        Args:
            denoising_label: int or None. The label of the processed image.

        Returns:
            bool.
        """
        if denoising_label is None:
            return False
        assert isinstance(denoising_label, int) or isinstance(denoising_label, np.int64)

        return denoising_label == self.__target_label

    def is_successful(self):
        """
        Returns denoising instance's status.
        Returns:
            bool.
        """
        return self._is_successful(self.denoising_label)

    def try_accept_the_example(self, denoising_sample, denoising_label):
        """
        If denoising_label is the target label that we are finding.
        The denoising_sample and denoising_label will be accepted and
        True will be returned.
        Else the denoising_sample will be stored in __bad_denoising_example.
        Args:
            denoising_sample: numpy.ndarray.
            denoising_label: int.

        Returns:
            bool.
        """
        assert isinstance(denoising_sample, np.ndarray)
        assert isinstance(denoising_label, int) or isinstance(denoising_label, np.int64)
        # assert self.input.shape == denoising_sample.shape

        ok = self._is_successful(denoising_label)
        if ok:
            self.__denoising_sample = denoising_sample
            self.denoising_label = denoising_label
        else:
            self.__bad_denoising_example = denoising_sample

        return ok

    def perturbation(self, multiplying_factor=1.0):
        """
        Compute perturbation between input and denoising_examples.
        Args:
            multiplying_factor: float.

        Returns:
            numpy.ndarray. The perturbation that is multiplied by multiplying_factor.
        """
        assert self.__input is not None
        assert (self.__denoising_sample is not None) or \
               (self.__bad_denoising_example is not None)
        if self.__denoising_sample is not None:
            return multiplying_factor * (
                self.__denoising_sample - self.__input)
        else:
            return multiplying_factor * (
                self.__bad_denoising_example - self.__input)

    @property
    def input(self):
        """
        :property: input image
        """
        return self.__input

    @property
    def target_label(self):
        """
        :property: target label
        """
        return self.__target_label

    @property
    def input_label(self):
        """
        property.
        Returns:
            input label.
        """
        return self.__input_label

    @property
    def denoising_sample(self):
        """
        :property: storing the denoising sample
        """
        return self.__denoising_sample

    @property
    def bad_denoising_example(self):
        """
        :property: storing the bad_denoising_example
        """
        return self.__bad_denoising_example
