#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
This module provides an example of Label-Only Membership Inference Attacks
ref paper: http://proceedings.mlr.press/v139/choquette-choo21a/choquette-choo21a.pdf
"""
from __future__ import print_function
import numpy as np
import sys
import os

import abc
import paddle

from typing import List
from paddle import Tensor
from .membership_inference_attack import MembershipInferenceAttack

class Classifier(paddle.nn.Layer):
    """
    Define a Classifier Network
    """
    def __init__(self, intput_size):
        """
        Init classifier class
        """
        super(Classifier, self).__init__()
        hidden_layer = 64
        self.input = paddle.nn.Linear(intput_size, hidden_layer)
        self.tanh = paddle.nn.Tanh()
        self.linear = paddle.nn.Linear(hidden_layer, hidden_layer)
        self.out = paddle.nn.Linear(hidden_layer, 2)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.input(x)
        y = self.linear(y)
        y = self.tanh(y)
        y = self.out(y)
        y = self.sigmoid(y)
        return y


class ComposeDataset(paddle.io.Dataset):
    """
    Data sets compose
    """
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list
        self.len = len(label_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label

    def __len__(self):
        return self.len

def create_rotates(r):
    """
    Creates vector of rotation degrees compatible with scipy' rotate.
    """
    if r is None:
      return None
    if r == 1:
      return [0.0]
    rotates = np.linspace(-r, r, (r * 2 + 1))
    return rotates

def check_correct(ds, predictions):
    """Used for augmentation MI attack to check if each image was correctly classified using label-only access.

    Args:
      ds: label.
      predictions: predictions from model.

    Returns: 1 if correct, 0 if incorrect for each sample.

    """
    return paddle.equal(ds, paddle.argmax(predictions, axis=1))

def augmentation_attack_set(model, train_set, test_set, batch_size, attack_type='r', augment_kwarg=1):
    """
    data augmentation and predict
    """
    if attack_type == 'r':
      augments = create_rotates(augment_kwarg)
    else:
      raise ValueError(f"attack type_: {attack_type} is not valid.")
    m = paddle.concat([paddle.ones([len(train_set)], dtype='int64'),
                        paddle.zeros([len(test_set)], dtype='int64')], axis=0)
    attack_in = paddle.zeros((len(train_set), len(augments)))
    attack_out = paddle.zeros((len(test_set), len(augments)))
    for i, augment in enumerate(augments):
      train_augment = []
      test_augment = []
      for j, data in  enumerate(train_set):
        train_augment.append((paddle.vision.transforms.rotate(data[0], augment), data[1]))
      for j, data in enumerate(test_set):
        test_augment.append((paddle.vision.transforms.rotate(data[0], augment), data[1]))
      train_dataset = ComposeDataset([x[0] for x in train_augment], [y[1] for y in train_augment])
      test_dataset = ComposeDataset([x[0] for x in test_augment], [y[1] for y in test_augment])
      train_loader = paddle.io.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
      test_loader = paddle.io.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
      
      in_ = model.predict(train_loader, batch_size=batch_size, stack_outputs=True)
      out_ = model.predict(test_loader, batch_size=batch_size, stack_outputs=True)
      softmax = paddle.nn.Softmax()
      in_ = softmax(paddle.to_tensor(in_[0]))
      out_ = softmax(paddle.to_tensor(out_[0]))
      train_label = paddle.to_tensor(np.array([y[1] for y in train_augment]).flatten())
      test_label = paddle.to_tensor(np.array([y[1] for y in test_augment]).flatten())
      attack_in[:, i] = paddle.cast(check_correct(train_label, in_)[:len(train_set)], "float32")
      attack_out[:, i] = paddle.cast(check_correct(test_label, out_)[:len(test_set)], "float32")
    attack_set = (paddle.concat([attack_in, attack_out], 0),
                  paddle.concat([paddle.to_tensor(np.array([y[1] for y in train_augment]).flatten()), 
                  paddle.to_tensor(np.array([y[1] for y in test_augment]).flatten())], 0),
                  m)
    return attack_set

class LabelOnlyMembershipInferenceAttack(MembershipInferenceAttack):
    """
    Label-Only membership inference attack class that
    utilizes shadow model and shadow data 
    """

    """
    Params:
    batch_size(int): The batch size for training shadow model and classifier
    shadow_epoch(int): The epochs for training shadow model
    classifier_epoch(int): The epochs for training classifier
    shadow_lr(float): The learning rate for training shadow model
    classifier_lr(float): The learning rate for training classifier
    attack_type(str): Type of attack to perform
    aug_kwarg(int): Param in rotation attack if used
    """
    params = ["batch_size", "shadow_epoch", "classifier_epoch",
              "shadow_lr", "classifier_lr", "attack_type", "aug_kwarg"]

    def __init__(self, shadow_model, shadow_dataset):
        """
        Init Label-only membership inference attack

        Args:
            shadow_model(Layer|Model): Shadow model for ML-Leaks.
            shadow_dataset(List[DataLoader|Dataset]): Datasets that used for training shadow model,
                including member-dataset and non-member dataset,
                that is shadow_dataset = [mem_data, non_mem_data]
        """
        self.shadow_model = shadow_model
        self.shadow_dataset = shadow_dataset
        self.classifier = None

        if isinstance(self.shadow_model, paddle.nn.Layer):
            self.shadow_model = paddle.Model(self.shadow_model)

        if isinstance(self.classifier, paddle.nn.Layer):
            self.classifier = paddle.Model(self.classifier)
            
    def set_params(self, **kwargs):
        """
        Set parameters for attacks

        Args:
            kwargs(dict): Parameters of dictionary type
        """
        super().set_params(**kwargs)
        self.__check_params()

        self._prepare()

    def infer(self, data, **kwargs) -> paddle.Tensor:
        """
        Infer whether data is in training set

        Args:
            data(Tensor): predict results, used to infer its membership (whether in training set)

        Returns:
            (Tensor): infer result
        """
        result = self.classifier.predict(data, batch_size=self.batch_size, stack_outputs=True)
        return paddle.to_tensor(result[0])

    def _prepare(self):
        """
        Train shadow model and classifier
        """
        input_size = 2 * self.aug_kwarg + 1
        if self.classifier is None:
            self.classifier = paddle.Model(Classifier(input_size))

        # train shadow model
        self.shadow_model.prepare(paddle.optimizer.Adam(parameters=self.shadow_model.parameters(),
                                                        learning_rate=self.shadow_lr),
                                    paddle.nn.CrossEntropyLoss(),
                                    [paddle.metric.Accuracy()])
        print("training shadow_model ...")
        shadow_train_data = paddle.io.DataLoader(self.shadow_dataset[0], shuffle=True, batch_size=self.batch_size)
        
        self.shadow_model.fit(shadow_train_data, epochs=self.shadow_epoch,
                                verbose=1, batch_size=self.batch_size)

        print("shadow model predict")

        #data augment
        self.shadow_dataset = augmentation_attack_set(self.shadow_model, 
                                                      self.shadow_dataset[0], 
                                                      self.shadow_dataset[1],
                                                      self.batch_size, 
                                                      self.attack_type, 
                                                      self.aug_kwarg)
        

        # train classifier
        self._train_classifier(self.shadow_model, self.classifier, self.shadow_dataset[0], self.shadow_dataset[2],
                          self.classifier_epoch, self.classifier_lr, self.batch_size)



    def _train_classifier(self, shadow_model, classifier, train_data_set, train_label_set,
                          epoch, learning_rate, batch_size):
        """
        Train classifier with predict results
        """
        compose_data = ComposeDataset(train_data_set, train_label_set)

        # 80% data for training, 20% data for testing
        train_len = int(len(train_label_set) * 4.0 / 5.0)
        test_len = len(train_label_set) - train_len
        train_data, test_data = paddle.io.random_split(compose_data, [train_len, test_len])
        train_loader = paddle.io.DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        test_loader = paddle.io.DataLoader(test_data, shuffle=True, batch_size=self.batch_size)
        # training classifier
        classifier.prepare(paddle.optimizer.Adam(learning_rate, parameters=classifier.parameters()),
                    paddle.nn.CrossEntropyLoss(),
                    [paddle.metric.Accuracy()])

        print("training classifier ...")
        classifier.fit(train_loader, test_loader, verbose=1, batch_size=batch_size, epochs=epoch)

    def __check_params(self) -> None:
        """
        Check params
        """
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The parameter of batch_size must be a positive int value.")

        if not isinstance(self.shadow_model, paddle.Model):
            raise ValueError("The parameter of shadow_model must paddle.Model value.")

        if (len(self.shadow_dataset) == 2) and (not
            isinstance(self.shadow_dataset[0], (paddle.io.DataLoader, paddle.io.Dataset))):
            raise ValueError("""The parameter of shadow_dataset must be a DataLoader or Dataset value.
                             And its length is 2""")

        if (not isinstance(self.shadow_epoch, int) or self.shadow_epoch <=0):
            raise ValueError("The parameter of shadow_epoch must be a positive int value.")

        if not isinstance(self.shadow_lr, float):
            raise ValueError("The parameter of shadow_lr must be a float value.")

        if not isinstance(self.classifier_epoch, int) or self.classifier_epoch <=0:
            raise ValueError("The parameter of classifier_epoch must be a positive int value.")

        if not isinstance(self.classifier_lr, float):
            raise ValueError("The parameter of classifier_lr must be a float value.")

        if not isinstance(self.attack_type, str):
            raise ValueError("The parameter of attack_type must be a string.")

        if not isinstance(self.aug_kwarg, int):
            raise ValueError("The parameter of attack_type must be a int value.")
