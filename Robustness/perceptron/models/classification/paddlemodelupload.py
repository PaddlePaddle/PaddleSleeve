# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""User classification model wrapper for Paddle."""

from __future__ import absolute_import

import os
import paddle
from perceptron.models.classification.paddle import PaddleModel


class PaModelUpload(PaddleModel):
    def __init__(self,
                 bounds,
                 num_classes,
                 channel_axis=1,
                 preprocessing=(0, 1)):
        # load model
        model = self.load_model()
        model.eval()

        super(PaModelUpload, self).__init__(model=model,
                                            bounds=bounds,
                                            num_classes=num_classes,
                                            channel_axis=channel_axis,
                                            preprocessing=preprocessing)

    @staticmethod
    def load_model():
        """load user model."""
        network = paddle.vision.models.resnet50(num_classes=10)
        model = paddle.Model(network)
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(here, '../../../examples/User_Model/checkpoint/test')
        print(model_path)
        model.load(model_path)
        model.network.eval()
        return model.network
