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
from __future__ import division
from builtins import object
from abc import ABCMeta
from abc import abstractmethod
import paddle
from future.utils import with_metaclass


class Model(with_metaclass(ABCMeta, object)):
    """
    Base class of model to provide attack.
    Args:
        bounds(tuple): (float, float). The range (lower and upper bound) for float value of the model input.
                    For normal distribution, we suggest set it as (-3, 3).
        channel_axis(int): The index of the axis that represents the color
                channel.
    """
    def __init__(self, bounds, channel_axis):
        assert len(bounds) == 2
        assert channel_axis in [0, 1, 2, 3]
        self.__bounds = bounds
        self.__channel_axis = channel_axis

    @property
    def bounds(self):
        """
        Return the upper and lower bounds of the model.
        """
        return self.__bounds

    @property
    def channel_axis(self):
        """
        Return the channel axis of the model.
        """
        return self.__channel_axis

    def _ensemble_models(self, model_list, model_weights):
        """
        Ensemble paddle2 models for inference.
        Args:
            model_list: A list of paddle2 models.s
            model_weights: A list of model weights.
        Returns:
            The one inference result of the ensemble paddle2 model.
        """
        class AnEnsembleModel(paddle.nn.Layer):
            """
            A paddle2 model wrapper that ensemble models in model list with given weights.
            """
            def __init__(self, _model_list, _model_weights):
                """
                Initialize needed variable.
                Args:
                    _model_list:
                    _model_weights:
                """
                super(AnEnsembleModel, self).__init__()
                self.model_list = _model_list
                self.model_weights = _model_weights
                self.weights_sum = sum(_model_weights)

            def forward(self, ipt):
                """
                Forward function for abstract method in paddle.nn.Layer
                Args:
                    ipt: paddle.Tensor input
                Returns:
                    Computed result in paddle.Tensor.
                """
                tmp_outputs = []
                for model in self.model_list:
                    tmp_outputs.append(model(ipt))
                out = 0
                for i, i_output in enumerate(tmp_outputs):
                    if self.weights_sum == 0:
                        i_model_weight = 0
                    else:
                        i_model_weight = self.model_weights[i] / self.weights_sum
                    out = out + i_model_weight * i_output
                return out
        weighted_ensemble_model = AnEnsembleModel(model_list, model_weights)
        return weighted_ensemble_model

    @abstractmethod
    def predict(self, data):
        """
        Calculate the prediction of the data.
        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).
        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """
        raise NotImplementedError

    @abstractmethod
    def num_classes(self):
        """
        Determine the number of the classes
        Return:
            int: the number of the classes
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, data, label):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.
        Args:
            data(numpy.ndarray): input data with shape (size, height, width,
            channels).
            label(int): Label used to calculate the gradient.
        Return:
            numpy.ndarray: gradient of the cross-entropy loss w.r.t the image
                with the shape (height, width, channel).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        raise NotImplementedError