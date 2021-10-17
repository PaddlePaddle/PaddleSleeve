# Copyright 2021 Baidu Inc.
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

"""Object detection model wrapper for pytorch hub models."""

from __future__ import absolute_import
from perceptron.models.base import Model
import torch

class YOLOv5sModel(Model):
    """Create a :class:`Model` instance from an `YOLOv5sModel` model.
    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        super(YOLOv5sModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._task = 'det'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


    def predictions(self, image):
        """Get prediction for input image
        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, w, c] ndarry format, BGR, 0 ~ 255
        Returns
        -------
        preditions.names stores the class name corresponding to the class label in .pred
        predicitons.pred[0]
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        """

        predictions = self.model(image)
        print("\n=====model predction: ", predictions.pred[0])
        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task
