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
from adversary import Adversary
from .base import AdvTransform
import numpy as np
import copy
import attacks


class ClassificationAdversarialTransform(AdvTransform):
    """
    This adversarial example generation module:
        * support adversarial sample generation from attack methods.
        * applying each attack method based on its probability given.
    """
    def __init__(self, paddlewhiteboxmodel, attacks, init_config_list, attack_config_list, **kwargs):
        """
        Args:
            paddlewhiteboxmodel: A PaddleWhiteBoxModel class instance.
            attack_methods: A list of implemented classes in <mother>.attacks.
            attack_config_list: A list of attack config corresponding to attack_methods.
            **kwargs: Other named arguments.
        """
        super(ClassificationAdversarialTransform, self).__init__(paddlewhiteboxmodel,
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
            return x, y
        else:
            adversary = Adversary(x.numpy(), y)
            # support the CW attack.
            # TODO: more generalizable.
            if isinstance(attack_method, attacks.cw.CWL2Attack):
                target_label = np.random.randint(self._paddlewhiteboxmodel.num_classes())
                while target_label == adversary.original_label:
                    target_label = np.random.randint(self._paddlewhiteboxmodel.num_classes())
                adversary.set_status(is_targeted_attack=True,
                                     target_label=target_label)
            else:
                pass
            adversary = attack_method(adversary, **tmp_config)
            if adversary.is_successful():
                adv_x = adversary.adversarial_example
                adv_y = y
            else:
                adv_x = adversary.original
                adv_y = y

        # TODO: try to return attack success record.
        return adv_x, adv_y
