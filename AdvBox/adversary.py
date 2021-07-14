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
It defines a class that contains the original object, the target and the
adversarial example.
"""
import numpy as np


class Adversary(object):
    """
    Usage philosophy:
    The objective of this tool box is to define a initial adversary instance and use
    attacks methods to transform it into a completed status (with a corresponding AE).

    In completed status, the Adversary.adversarial_example() is not None.
    """
    def __init__(self, original, original_label=None):
        """
        Initialization for Adversary object.
        Args:
            original: numpy.ndarray. The original sample, such as an image. Numpy
            original_label: numpy.ndarray. The original sample's label.
        """
        assert isinstance(original, np.ndarray)
        assert isinstance(original_label, int) or isinstance(original_label, np.int64)
        self.__original = original
        self.__original_label = original_label

        self.__is_targeted_attack = False
        self.__target_label = None
        self.adversarial_label = None

        self.__denormalized_original = None
        self.__denormalized_adversarial_example = None
        self.__denormalized_bad_adversarial_example = None

        self.__adversarial_example = None
        self.__bad_adversarial_example = None

    def summary(self):
        print("original label:", self.__original_label)
        print("target label:", self.__target_label)
        print("adversarial label:", self.adversarial_label)
        print("contains a successful AE:", self.is_successful())

    def set_status(self, is_targeted_attack, target_label=None):
        """
        Set the adversary instance to be:
        * targeted status.
        * untargeted status.
        Args:
            is_targeted_attack: bool. The flag for attack purpose type.
            target_label: int. If is_targeted_attack is true and target_label is
                        None, self.target_label will be set by the Attack class.
                        If is_targeted_attack is false, target_label should be None.
        Returns:
            None.
        """
        assert isinstance(is_targeted_attack, bool)
        assert isinstance(target_label, int) or isinstance(target_label, np.int64)
        self.__is_targeted_attack = is_targeted_attack
        if self.__is_targeted_attack:
            # targeted status
            assert target_label is not None
            self.__target_label = target_label
        else:
            # untargeted status
            self.__target_label = None

    def reset(self):
        """
        Reset adversary status.
        Returns:
            None.
        """
        self.__is_targeted_attack = False
        self.__target_label = None
        self.adversarial_label = None

        self.__denormalized_original = None
        self.__denormalized_adversarial_example = None
        self.__denormalized_bad_adversarial_example = None

        self.__adversarial_example = None
        self.__bad_adversarial_example = None

    def _is_successful(self, adversarial_label):
        """
        Check if the adversarial_label is the expected adversarial label.
        Args:
            adversarial_label: int or None. adversarial label.

        Returns:
            bool.
        """
        if adversarial_label is None:
            return False
        assert isinstance(adversarial_label, int) or isinstance(adversarial_label, np.int64)
        if self.__is_targeted_attack:
            return adversarial_label == self.__target_label
        else:
            return adversarial_label != self.__original_label

    def is_successful(self):
        """
        Returns adversary instance's status.
        Returns:
            bool.
        """
        return self._is_successful(self.adversarial_label)

    def try_accept_the_example(self, adversarial_example, adversarial_label):
        """
        If adversarial_label the target label that we are finding.
        The adversarial_example and adversarial_label will be accepted and
        True will be returned.
        Else the adversarial_example will be stored in __bad_adversarial_example.
        Args:
            adversarial_example: numpy.ndarray.
            adversarial_label: int.

        Returns:
            bool.
        """
        assert isinstance(adversarial_example, np.ndarray)
        assert isinstance(adversarial_label, int) or isinstance(adversarial_label, np.int64)
        assert self.original.shape == adversarial_example.shape

        ok = self._is_successful(adversarial_label)
        if ok:
            self.__adversarial_example = adversarial_example
            self.adversarial_label = adversarial_label
        else:
            self.__bad_adversarial_example = adversarial_example

        return ok

    def perturbation(self, multiplying_factor=1.0):
        """
        Compute perturbation between original and adversary_examples.
        Args:
            multiplying_factor: float.

        Returns:
            numpy.ndarray. The perturbation that is multiplied by multiplying_factor.
        """
        assert self.__original is not None
        assert (self.__adversarial_example is not None) or \
               (self.__bad_adversarial_example is not None)
        if self.__adversarial_example is not None:
            return multiplying_factor * (
                self.__adversarial_example - self.__original)
        else:
            return multiplying_factor * (
                self.__bad_adversarial_example - self.__original)

    @property
    def is_targeted_attack(self):
        """
        :property: is_targeted_attack
        """
        return self.__is_targeted_attack

    @property
    def original(self):
        """
        :property: original
        """
        return self.__original

    @property
    def target_label(self):
        """
        :property: target
        """
        return self.__target_label

    @property
    def original_label(self):
        """
        property.
        Returns:
            original label.
        """
        return self.__original_label

    @property
    def adversarial_example(self):
        """
        :property: adversarial_example
        """
        return self.__adversarial_example

    @property
    def bad_adversarial_example(self):
        """
        :property: bad_adversarial_example
        """
        return self.__bad_adversarial_example
