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
Paddle Model
"""
from __future__ import absolute_import
from __future__ import print_function
from .base import Model
import paddle
import logging
logger = logging.getLogger(__name__)


class PaddleWhiteBoxModel(Model):
    """
    paddle white box model
    * support adversarial sample generation based on weighted multi-model ensemble attack.
    """
    def __init__(self,
                 model_list,
                 model_weights,
                 loss=None,
                 bounds=None,
                 channel_axis=3,
                 nb_classes=1000):
        """
        Paddle model for white box attack.
        Args:
            model_list: List. A list of Paddle2 model.
            model_weights: List. A list of float weights for each model to consider.
            loss: Paddle.Op. Loss function for supervised classification.
            bounds(tuple): (float, float). The range (lower and upper bound) for float value of the model input.
                For normal distribution, we suggest set it as (-3, 3).
            channel_axis(int): The index of the axis that represents the color
                channel.
            nb_classes: int. number of classification class.
        """
        assert len(model_list) == len(model_weights)
        super(PaddleWhiteBoxModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis)
        self._model_list = model_list
        self._model_weights = model_weights
        self._weights_sum = sum(model_weights)
        self._weighted_ensemble_model = self._ensemble_models(model_list, model_weights)
        self._loss = loss

        # check if nb_classes is correct by probing model and see its output
        probe_inputdata = paddle.ones((1, channel_axis, 1, 1))
        probe_output = self.predict_tensor(probe_inputdata)
        assert probe_output.shape[1] == nb_classes
        self._nb_classes = nb_classes

        self._device = paddle.get_device()
        print("Paddle Device: ", self._device)
        logger.info("Finished PaddleWhiteBoxModel Initialization")

    def predict(self, data):
        """
        Calculate the prediction of the data.
        Args:
            data: Numpy.ndarray Input data with shape (size, height, width, channels).
        Return:
            numpy.ndarray: Predictions of the data with shape (batch_size, num_of_classes).
        """
        # freeze BN when forwarding
        for model in self._model_list:
            for param in model.parameters():
                param.stop_gradient = True
            for module in model.sublayers():
                if isinstance(module, (paddle.nn.BatchNorm, paddle.nn.BatchNorm1D,
                                       paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)):
                    # print("evaled!!")
                    module.eval()

        tensor_data = paddle.to_tensor(data, dtype='float32', place=self._device)
        # Run prediction
        predict = self._weighted_ensemble_model(tensor_data)

        # free model parameter
        for model in self._model_list:
            for param in model.parameters():
                param.stop_gradient = False
            for module in model.sublayers():
                if isinstance(module, (paddle.nn.BatchNorm, paddle.nn.BatchNorm1D,
                                       paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)):
                    # print("trained!!")
                    module.train()
        return predict.numpy()

    def predict_tensor(self, data):
        """
        Calculate the prediction of the data. Usually used for compute grad for input.
        Args:
            data: Paddle.Tensor input data with shape (size, height, width, channels).
        Return:
            Paddle.Tensor: predictions of the data with shape (batch_size, num_of_classes).
        """
        # freeze BN when forwarding
        for model in self._model_list:
            for param in model.parameters():
                param.stop_gradient = True
            for module in model.sublayers():
                if isinstance(module, (paddle.nn.BatchNorm, paddle.nn.BatchNorm1D,
                                       paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)):
                    # print("evaled!!")
                    module.eval()

        # Run prediction
        predict = self._weighted_ensemble_model(data)

        # free model parameter
        for model in self._model_list:
            for param in model.parameters():
                param.stop_gradient = False
            for module in model.sublayers():
                if isinstance(module, (paddle.nn.BatchNorm, paddle.nn.BatchNorm1D,
                                       paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)):
                    # print("trained!!")
                    module.train()
        return predict

    def num_classes(self):
        """
        Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """
        return self._nb_classes

    def gradient(self, data, label):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.
        Args:
            data: Numpy.ndarray input with shape as (size, height, width, channels).
            label: Int used to compute the gradient. When ensemble multi-models, keep labels consistent for all models.
        Return:
            numpy.ndarray. gradient of the cross-entropy loss w.r.t the image
                with the shape (height, width, channel).
        """
        tensor_data = paddle.to_tensor(data, dtype='float32', place=self._device)
        tensor_data.stop_gradient = False
        label = paddle.to_tensor(label, dtype='int64', place=self._device)
        output = self.predict_tensor(tensor_data)
        loss = self._loss(output, label)
        loss.backward(retain_graph=True)
        grad = tensor_data.grad.numpy()

        return grad

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        Returns:
            string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
