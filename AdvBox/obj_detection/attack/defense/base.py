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
The base logic for adversarial data augmentation.
"""
import logging
import numpy as np
import paddle
from abc import ABCMeta
from abc import abstractmethod

from paddle.fluid.dataloader.collate import default_collate_fn
from attack.utils.tools import get_distance, get_criteria, denormalize_image


class AdvTransform(object):
    """
    Abstract base class for adversarial examples generation based on Paddle2
    model and Paddle adversarial attacks. Subclass should implement the
    _generate_adv_example(self, model_list, x, y) method.
    """
    def __init__(self, model, attacks, init_config_list, attack_config_list, **kwargs):
        """
        Args:
            model: A PaddleWhiteBoxModel class instance.
            attack_methods: A list of implemented classes in <mother>.attacks.
            init_config_list: A list of attack method init config.
            attack_config_list: A list of attack config corresponding to attack_methods.
            **kwargs: Other named arguments.
        """
        np.random.seed(0)
        self._model = model
        self._attack_methods = [method for method in attacks]
        self._attack_probability = [config.pop("p") for config in attack_config_list]
        self._init_config_list = init_config_list
        self._attack_config_list = attack_config_list
        assert sum(self._attack_probability) <= 1
        assert sum(self._attack_probability) >= 0
        # if the sum p value in config does not add up to 1, the rest probability will be
        # pass through choice
        self._attack_probability.append(1 - sum(self._attack_probability))
        self._attack_instances = []

        for i, attack_method in enumerate(self._attack_methods):
            if attack_method is not None:
                if self._init_config_list[i] is None:
                    self._attack_instances.append(attack_method(self._model, verbose=False))
                else:
                    distance = get_distance(self._init_config_list[i].pop('distance', 'l2'))
                    criteria = get_criteria('target_class_miss', target_class=-1, model_name='paddledet_')
                    self._attack_instances.append(attack_method(self._model, criterion=criteria, distance=distance, verbose=False))
            else:
                pass

    def __call__(self, x_batch):
        """
        Transform x to adv_x.
        Args:
            x_batch: A list of dict, each corresponds to one input sample
            y_batch: A list of dict, each corresponds to the grond truth of one sample
        Returns:
            Transformed adversarial examples, collated in batch
        """
        for x in x_batch:
            target = x['target_class'].numpy() if 'target_class' in x else 0
            attack_method, attack_config = self._choose_attack()
            if attack_method is None:
                continue
            attack_method._default_model._data = default_collate_fn([x])
            ori_img = denormalize_image(paddle.squeeze(x['image']), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            adv_x = self._generate_adv_example(ori_img.numpy(), target, attack_method, attack_config)
            adv_x = self._model._preprocessing(paddle.to_tensor(adv_x, dtype='float32'))
            x['image'] = adv_x
        return x_batch

    def _choose_attack(self):
        choices_num = len(self._attack_methods)
        # range(0, choices_num + 1) is the index for pass through choice
        choosed_index = np.random.choice(range(0, choices_num + 1), p=self._attack_probability)
        if choosed_index == choices_num:
            current_attack_method = None
            attack_config = None
        else:
            current_attack_method = self._attack_instances[choosed_index]
            attack_config = self._attack_config_list[choosed_index]
        return current_attack_method, attack_config

    @abstractmethod
    def _generate_adv_example(self, x, y, attack_method, attack_config):
        """
        Generate adv sample based on ensemble of models.
        Args:
            x: An image sample.
            y: An image label.
            attack_method: An attack instantce.
            attack_config: A adv image sample and its original label.
        Returns:
            adv_x, adv_y
        """
        raise NotImplementedError

