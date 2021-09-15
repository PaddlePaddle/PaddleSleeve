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

"""Image classification model wrapper for paddle hub models."""

from __future__ import absolute_import
from perceptron.models.base import Model
import paddlehub as hub
import numpy as np

class MobilenetV2AnimalsModel(Model):
    """Create a :class:`Model` instance from an `MobulenetV2AnimalsModel` model.
    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        super(MobilenetV2AnimalsModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._task = 'cls'
        self.model = hub.Module(name="mobilenet_v2_animals")

    def predictions(self, image):
        """Get prediction for input image
        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, w, c] ndarry format, BGR, 0 ~ 255
        Returns
        -------
        """
        image = image.astype(np.uint8)
        predictions = self.model.classification(images=[image])
        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task