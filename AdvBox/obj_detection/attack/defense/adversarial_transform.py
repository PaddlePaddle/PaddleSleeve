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
Adversarial Transform tool based on attacks and paddle models
"""
from __future__ import absolute_import

import os
import sys

from attack.defense.base import AdvTransform
import copy
import gc

work_dir = os.path.abspath(os.getcwd())
sys.path.append(work_dir)


class DetectionAdversarialTransform(AdvTransform):
    """
    This adversarial example generation module:
        * support adversarial sample generation from attack methods.
        * applying each attack method based on its probability given.
    """
    def __init__(self, model, attacks, init_config_list, attack_config_list, **kwargs):
        """
        Args:
            model (paddle.nn.Layers): A paddle model
            attack_methods: A list of implemented classes in <mother>.attacks.
            attack_config_list: A list of attack config corresponding to attack_methods.
            **kwargs: Other named arguments.
        """
        super(DetectionAdversarialTransform, self).__init__(model,
                                                            attacks,
                                                            init_config_list,
                                                            attack_config_list)

    def _generate_adv_example(self, x, y, attack_method, attack_config):
        """
        A logic for generate adversarial perturbation for x, y.
        Args:
            x: numpy.ndarray. original input sample.
            y: numpy.int or int. original input label.
            attack_method: Attack.
            attack_config: Dict.
        Returns:
            adv_x, adv_y
        """
        tmp_config = copy.deepcopy(attack_config)
        if attack_method is None:
            return x
        else:
            attack_method._default_criterion._target_class = y
            adversary = attack_method(x, y, unpack=False, **tmp_config)

            if adversary.image is not None:
                adv_x = adversary.image
            else:
                adv_x = adversary.original_image
        del adversary
        gc.collect()
        return adv_x
