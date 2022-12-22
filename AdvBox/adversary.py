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
# TODO: support batch input
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
            original: numpy.ndarray. The single original sample(no batch), such as an image.
            original_label: numpy.ndarray. The original sample's label.
        """
        assert isinstance(original, np.ndarray)
        assert isinstance(original_label, int) or isinstance(original_label, np.int64)
        self._original = original
        self._original_label = original_label

        self._is_targeted_attack = False
        self._target_label = None

        self._denormalized_original = None
        self._denormalized_adversarial_example = None
        self._denormalized_bad_adversarial_example = None

        self._adversarial_label = None
        self._adversarial_example = None
        self._bad_adversarial_example = None

        self._sample_channel_num = None

    def summary(self):
        print("original label:", self.original_label)
        print("target mode:", self.is_targeted_attack)
        print("target label:", self.target_label)
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
        self._is_targeted_attack = is_targeted_attack
        if self._is_targeted_attack:
            # targeted status
            self._target_label = target_label
        else:
            # untargeted status
            self._target_label = None

    def routine_check(self, advbox_base_model):
        """
        Check model property is consistent with adversary property.
        Args:
            advbox_base_model: an instance of a models.base.Model

        Returns:
            None
        """
        # make sure model property is consistent with dataset
        assert self.original.shape == advbox_base_model.input_shape
        self._sample_channel_num = self.original.shape[advbox_base_model.input_channel_axis]

    def generate_denormalized_original(self, input_channel_axis, mean, std):
        """
        Denormalize input sample with given mean & std if given.
        We use denormalized original for perturbation process.
        Args:
            input_channel_axis: int. the channel index number of input sample.
            mean: list. channelwise average values.
            std: list. channelwise standard deviation values.

        Returns:
            None
        """
        assert self.original.shape[input_channel_axis] == len(mean)

        self._denormalized_original = np.zeros(self.original.shape)
        for channel in range(self.original.shape[input_channel_axis]):
            self._denormalized_original[channel] = self.original[channel] * std[channel] + mean[channel]

    # TODO: leave it as it is or delete?
    #  when adding new attack and implementing the _apply, the graceful way is to try to
    #  accept denormalized_adversarial_example & adversarial_example simultaneously?
    # def generate_normalized_adversarial_example(self, input_channel_axis, mean, std):
    #     """
    #     Normalize generated adversarial sample from denormalized domain.
    #     Args:
    #         input_channel_axis: int. the channel index number of input sample.
    #         mean: list. channelwise average values.
    #         std: list. channelwise standard deviation values.
    #     Returns:
    #         None
    #     """
    #     assert self.original.shape[input_channel_axis] == len(mean)
    #
    #     ok = self.is_successful()
    #     if ok:
    #         self._adversarial_example = np.zeros(self.original.shape)
    #         for channel in range(self.original.shape[input_channel_axis]):
    #             self._adversarial_example[channel] = \
    #                 (self._denormalized_adversarial_example[channel] - mean[channel]) / std[channel]
    #     else:
    #         self._bad_adversarial_example = np.zeros(self.original.shape)
    #         for channel in range(self.original.shape[input_channel_axis]):
    #             self._bad_adversarial_example[channel] = \
    #                 (self._denormalized_bad_adversarial_example[channel] - mean[channel]) / std[channel]

    def reset(self):
        """
        Reset adversary status.
        Returns:
            None.
        """
        self._is_targeted_attack = False
        self._target_label = None

        self._denormalized_original = None
        self._denormalized_adversarial_example = None
        self._denormalized_bad_adversarial_example = None

        self._adversarial_label = None
        self._adversarial_example = None
        self._bad_adversarial_example = None

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
        if self._is_targeted_attack:
            #return adversarial_label == self._target_label
            return adversarial_label != self._original_label
        else:
            return adversarial_label != self._original_label

    def is_successful(self):
        """
        Returns adversary instance's status.
        Returns:
            bool.
        """
        return self._is_successful(self._adversarial_label)

    def try_accept_the_example(self, denormalized_adversarial_example, adversarial_example, adversarial_label):
        """
        If adversarial_label the target label that we are finding.
        The adversarial_example and adversarial_label will be accepted and
        True will be returned.
        Else the adversarial_example will be stored in _bad_adversarial_example.
        Args:
            denormalized_adversarial_example: numpy.ndarray.
            adversarial_example: numpy.ndarray.
            adversarial_label: int.

        Returns:
            bool.
        """
        assert isinstance(denormalized_adversarial_example, np.ndarray)
        assert isinstance(adversarial_example, np.ndarray)
        assert isinstance(adversarial_label, int) or isinstance(adversarial_label, np.int64)
        assert self.denormalized_original.shape == adversarial_example.shape
        assert self.denormalized_original.shape == denormalized_adversarial_example.shape

        ok = self._is_successful(adversarial_label)
        if ok:
            self._adversarial_example = adversarial_example
            self._denormalized_adversarial_example = denormalized_adversarial_example
            self._adversarial_label = adversarial_label
        else:
            self._bad_adversarial_example = adversarial_example
            self._denormalized_bad_adversarial_example = denormalized_adversarial_example

        return ok

    def perturbation(self, multiplying_factor=1.0):
        """
        Compute perturbation between original and adversary_examples.
        Args:
            multiplying_factor: float.

        Returns:
            numpy.ndarray. The perturbation that is multiplied by multiplying_factor.
        """
        assert self._original is not None
        assert (self._adversarial_example is not None) or \
               (self._bad_adversarial_example is not None)
        if self._adversarial_example is not None:
            return multiplying_factor * (
                self._adversarial_example - self._original)
        else:
            return multiplying_factor * (
                self._bad_adversarial_example - self._original)

    @property
    def original(self):
        """
        :property: original
        """
        return self._original

    @property
    def original_label(self):
        """
        property.
        Returns:
            original label.
        """
        return self._original_label

    @property
    def is_targeted_attack(self):
        """
        :property: is_targeted_attack
        """
        return self._is_targeted_attack

    @property
    def target_label(self):
        """
        :property: target
        """
        return self._target_label

    @property
    def denormalized_original(self):
        """
        :property: denormalized original
        """
        return self._denormalized_original

    @property
    def denormalized_adversarial_example(self):
        """
        :property: denormalized adversarial_example
        """
        return self._denormalized_adversarial_example

    @property
    def adversarial_example(self):
        """
        :property: adversarial_example
        """
        return self._adversarial_example

    @property
    def adversarial_label(self):
        """
        :property: adversarial_label
        """
        return self._adversarial_label

    @property
    def denormalized_bad_adversarial_example(self):
        """
        :property: denormalized bad adversarial example
        """
        return self._denormalized_bad_adversarial_example

    @property
    def bad_adversarial_example(self):
        """
        :property: bad_adversarial_example
        """
        return self._bad_adversarial_example

    @property
    def sample_channel_num(self):
        """
          :property: sample_channel_num
        """
        return self._sample_channel_num
