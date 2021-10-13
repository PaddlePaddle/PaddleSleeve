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

"""Keras model wrapper for SSD300 object detection."""

from __future__ import absolute_import
import numpy as np
from perceptron.models.base import DifferentiableModel


class KerasSSD300Model(DifferentiableModel):
    """Create a :class:`Model` instance from a `DifferentiableModel` model.
    Parameters
    ----------
    model : `keras.model.Model`
        The `Keras` model that are loaded.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    model_image_shape : tuple
        Tuple of the model input shape in format (height, width).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    max_boxes : int
        The maximum number of boxes allowed in the prediction output.
    score : float
        The score threshold for considering a box as containing objects.
    iou : float
        The intersection over union (IoU) threshold.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.
    class_names : list
        Class names for ground truth labels
    """

    def __init__(self,
                 model,
                 bounds,
                 model_image_shape=(300, 300),
                 num_classes=20,
                 channel_axis=3,
                 max_boxes=20,
                 score=0.2,
                 iou=0.5,
                 preprocessing=(0, 1),
                 class_names=['background',
                              'aeroplane', 'bicycle', 'bird', 'boat',
                              'bottle', 'bus', 'car', 'cat',
                              'chair', 'cow', 'diningtable', 'dog',
                              'horse', 'motorbike', 'person', 'pottedplant',
                              'sheep', 'sofa', 'train', 'tvmonitor']):
        super(KerasSSD300Model, self).__init__(bounds=bounds,
                                               channel_axis=channel_axis,
                                               preprocessing=preprocessing)

        self._model = model
        self._th_conf = score
        self._min_overlap = iou
        self._img_height = model_image_shape[0]
        self._img_width = model_image_shape[1]
        self._class_names = class_names
        self._num_classes = num_classes
        self._task = 'det'

    def num_classes(self):
        """Return the number of classes."""
        return self._num_classes

    def class_names(self):
        """Return the class names as list."""
        return self._class_names

    def get_class(self):
        return self.class_names()

    def model_task(self):
        """Return the task of the model: classification of detection."""
        return self._task

    def batch_predictions(self, images):
        """Batch prediction of images.
        Parameters
        ----------
        images : `numpy.ndarray`
            The input image in [b, h, w, c] ndarry format.
        Returns
        -------
        list
            List of batch prediction resutls.
            Each element is a dictionary containing:
            {'boxes', 'scores', 'classes}
        """
        import cv2
        images_res = []
        for image in images:
            image_res = cv2.resize(image, (self._img_height, self._img_width))
            images_res.append(image_res)
        images_res = np.array(images_res)
        y_preds = self._model.predict(images_res)
        results = []
        for y_pred in y_preds:
            result = {}
            out_boxes = []
            out_scores = []
            out_classes = []
            for temp_pred in y_pred:
                if temp_pred[1] >= self._th_conf:
                    temp_bbox = temp_pred[2:]
                    temp_bbox = np.array(
                        [temp_bbox[1], temp_bbox[0], temp_bbox[3], temp_bbox[2]])
                    out_boxes.append(temp_bbox)
                    out_scores.append(temp_pred[1])
                    out_classes.append(int(temp_pred[0]))
                result['boxes'] = out_boxes
                result['scores'] = out_scores
                result['classes'] = out_classes
            results.append(result)

        return results

    def predictions_and_gradient(self, image, criterion):
        """Returns both predictions and gradients, and
        potentially loss w.r.t. to certain criterion.
        """
        pass

    def backward(self, target_class, image):
        """Get gradient with respect to the image."""
        pass
