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
The base model of the model.
"""
# TODO: support batch input.
import logging
from abc import ABCMeta
from abc import abstractmethod
import paddle


class Attack(object):
    """
    Abstract base class for adversarial attacks. `Attack` represent an
    adversarial attack which search an adversarial example. Subclass should
    implement the _apply(self, adversary, **kwargs) method.
    Args:
        model(Model): an instance of a models.base.Model.
        norm(str): 'Linf' or 'L2', the norm of the threat model.
        epsilon_ball(float): the bound on the norm of the AE.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, norm='Linf', epsilon_ball=8/255, epsilon_stepsize=2/255):
        # norm='L2', epsilon_ball=128/255, epsilon_stepsize=15/255
        self.model = model
        self._device = paddle.get_device()
        assert norm in ('Linf', 'L2')
        self.norm = norm
        self.epsilon_ball = epsilon_ball
        self.epsilon_stepsize = epsilon_stepsize
        self.normalize = paddle.vision.transforms.Normalize(mean=self.model.normalization_mean,
                                                            std=self.model.normalization_std)

    def __call__(self, adversary, **kwargs):
        """
        Generate the adversarial sample.
        Args:
            adversary(object): The adversary object.
            **kwargs: Other named arguments.
        """
        # make sure data in adversary is compatible with self.model
        adversary.routine_check(self.model)
        adversary.generate_denormalized_original(self.model.input_channel_axis,
                                                 self.model.normalization_mean,
                                                 self.model.normalization_std)
        # _apply generate denormalized AE to perturb adversarial in pre-normalized domain
        adversary = self._apply(adversary, **kwargs)
        return adversary

    def input_preprocess(self, img):
        """
        Normalize img and add batchsize dimension safely.
        Args:
            img: paddle.tensor. initial input before normalization.

        Returns:
            unaked_img_normalized: paddle.tensor. normalized img that covered with batchsize dimension.
        """
        assert isinstance(img, paddle.Tensor)
        img_normalized = self.normalize(img)
        if len(self.model.input_shape) < img_normalized.ndim:
            unaked_img_normalized = img_normalized
        else:
            unaked_img_normalized = paddle.unsqueeze(img_normalized, axis=0)

        return unaked_img_normalized

    def safe_delete_batchsize_dimension(self, img):
        """
        Compare dimension setting in model, delete batchsize,
        dimension (axis=0) if there's dimension redundancy.
        Args:
            img: paddle.tensor

        Returns:
            naked_img: paddle.tensor. img that deleted redundant batchsize dimension.
        """
        assert isinstance(img, paddle.Tensor)
        if len(self.model.input_shape) < img.ndim:
            naked_img = paddle.squeeze(img, axis=0)
        else:
            naked_img = img

        return naked_img

    @abstractmethod
    def _apply(self, adversary, **kwargs):
        """
        Search an adversarial example.
        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.

        retun
        """
        raise NotImplementedError
