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
Implement of ML-Leaks membership inference attacks
ref paper: https://arxiv.org/pdf/1806.01246.pdf
"""


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
        self.len = [0]
        for data in self.data_list:
            self.len.append(self.len[len(self.len) - 1] + data.shape[0])
    
    def __getitem__(self, idx):
        index = 0
        for id, l in enumerate(self.len):
            if l > idx:
                index = id - 1
                break
            index = id
        
        data = self.data_list[index][idx - self.len[index]]
        label = self.label_list[index][idx - self.len[index]]
        return data, label

    def __len__(self):
        return self.len[len(self.len) - 1]


class MLLeaksMembershipInferenceAttack(MembershipInferenceAttack):
    """ 
    ML-Leaks membership inference attack class that
    utilizes shadow model and shadow data
    """

    """
    Params:
        batch_size(int): The batch size for training shadow model and classifier
        shadow_epoch(int): The epochs for training shadow model
        classifier_epoch(int): The epochs for training classifier
        shadow_lr(float): The learning rate for training shadow model
        classifier_lr(float): The learning rate for training classifier
        topk(int): The top k predict posteriors that used to train classifier
    """
    params = ["batch_size", "shadow_epoch", "classifier_epoch",
              "shadow_lr", "classifier_lr", "topk"]

    def __init__(self, shadow_model, shadow_dataset):
        """
        Init ML-Leaks membership inference attack

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
        data_k, _ = paddle.topk(data, self.topk)
        result = self.classifier.predict(data_k, batch_size=self.batch_size, stack_outputs=True)
        return paddle.to_tensor(result[0])

    def _prepare(self):
        """
        Train shadow model and classifier
        """
        if self.classifier is None:
            self.classifier = paddle.Model(Classifier(self.topk))

        # train shadow model
        self.shadow_model.prepare(paddle.optimizer.Adam(parameters=self.shadow_model.parameters(),
                                                        learning_rate=self.shadow_lr),
                                    paddle.nn.CrossEntropyLoss(),
                                    [paddle.metric.Accuracy()])
        print("training shadow_model ...")
        self.shadow_model.fit(self.shadow_dataset[0], self.shadow_dataset[1], epochs=self.shadow_epoch,
                                verbose=1, batch_size=self.batch_size)

        # train classifier
        self._train_classifier(self.shadow_model, self.classifier, self.shadow_dataset[0], self.shadow_dataset[1],
                          self.classifier_epoch, self.classifier_lr, self.topk, self.batch_size)


    def _train_classifier(self, shadow_model, classifier, shadow_train_data, shadow_test_data,
                          epoch, learning_rate, topk, batch_size):
        """
        Train classifier with predict results
        """
        train_data_set = shadow_model.predict(shadow_train_data, batch_size=batch_size, stack_outputs=True)
        test_data_set = shadow_model.predict(shadow_test_data, batch_size=batch_size, stack_outputs=True)

        train_data_set, _ = paddle.topk(paddle.to_tensor(train_data_set[0]), topk)
        train_label_size = train_data_set.shape[0]
        train_label_set = paddle.ones([train_label_size], dtype='int64')

        test_data_set, _ = paddle.topk(paddle.to_tensor(test_data_set[0]), topk)
        test_label_size = test_data_set.shape[0]
        test_label_set = paddle.zeros([test_label_size], dtype='int64')

        compose_data = ComposeDataset([train_data_set, test_data_set], [train_label_set, test_label_set])

        # 80% data for training, 20% data for testing
        train_len = int(len(compose_data) * 4.0 / 5.0)
        test_len = len(compose_data) - train_len
        train_data, test_data = paddle.io.random_split(compose_data, [train_len, test_len])

        # weighted based data sample, for unbalance dataset
        weights, large_label_count = self._cal_weight(train_data)
        sample_weight = paddle.io.WeightedRandomSampler(weights, num_samples=large_label_count * 2)
        batch_sample = paddle.io.BatchSampler(sampler=sample_weight, batch_size=batch_size, drop_last=True)
        train_loader = paddle.io.DataLoader(train_data, batch_sampler=batch_sample)

        # training classifier
        classifier.prepare(paddle.optimizer.Adam(learning_rate, parameters=classifier.parameters()),
                    paddle.nn.CrossEntropyLoss(),
                    [paddle.metric.Accuracy()])

        print("training classifier ...")

        callbacks = paddle.callbacks.EarlyStopping(
                                                'loss',
                                                mode='auto',
                                                patience=2,
                                                verbose=1,
                                                min_delta=0.01,
                                                baseline=None,
                                                save_best_model=True)

        classifier.fit(train_loader, test_data, verbose=1, batch_size=batch_size, epochs=epoch, callbacks=[callbacks])

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

        if not isinstance(self.topk, int):
            raise ValueError("The parameter of topk must be a positive int value.")
        

    def _cal_weight(self, dataset):
        """
        Calc sample weight, used for weighted sample
        """
        data_len = len(dataset)
        count_pos = 0
        for i in range(data_len):
            if dataset[i][1] == 1:
                count_pos += 1
        count_neg = data_len - count_pos
        pos_weight = 0.1
        neg_weight = 0.1 * float(count_pos) / count_neg
        weight = []
        for i in range(data_len):
            if dataset[i][1] == 1:
                weight.append(pos_weight)
            else:
                weight.append(neg_weight)
        return weight, count_neg if count_neg > count_pos else count_pos

