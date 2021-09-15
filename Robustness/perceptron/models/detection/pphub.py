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


class PedestrianDetModel(Model):
    """Create a :class:`Model` instance from an `PedestrianDetModel` model.
    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.
    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        super(PedestrianDetModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._task = 'det'
        self.model = hub.Module(name="yolov3_darknet53_pedestrian")

    def predictions(self, image):
        """Get prediction for input image
        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, w, c] ndarry format, BGR, 0 ~ 255
        Returns
        -------
        res (list[dict]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
            data (list): 检测结果，list的每一个元素为 dict，各字段为:
                confidence (float): 识别的置信度；
                label (str): 标签；
                left (int): 边界框的左上角x坐标；
                top (int): 边界框的左上角y坐标；
                right (int): 边界框的右下角x坐标；
                bottom (int): 边界框的右下角y坐标；
            save_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)。


        def object_detection(paths=None,
                             images=None,
                             batch_size=1,
                             use_gpu=False,
                             output_dir='detection_result',
                             score_thresh=0.2,
                             visualization=True)
        """

        predictions = self.model.object_detection(images=[image])
        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task


class VehiclesDetModel(Model):
    """Create a :class:`Model` instance from an `VehiclesDetModel` model.
    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.
    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        super(VehiclesDetModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._task = 'det'
        self.model = hub.Module(name="yolov3_darknet53_vehicles")

    def predictions(self, image):
        """Get prediction for input image
        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, w, c] ndarry format, BGR, 0 ~ 255
        Returns
        -------
        res (list[dict]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
            data (list): 检测结果，list的每一个元素为 dict，各字段为:
                confidence (float): 识别的置信度；
                label (str): 标签；
                left (int): 边界框的左上角x坐标；
                top (int): 边界框的左上角y坐标；
                right (int): 边界框的右下角x坐标；
                bottom (int): 边界框的右下角y坐标；
            save_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)。


        def object_detection(paths=None,
                             images=None,
                             batch_size=1,
                             use_gpu=False,
                             output_dir='detection_result',
                             score_thresh=0.5,
                             visualization=True)
        """
        predictions = self.model.object_detection(images=[image])
        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task


class SSDVGG16300Model(Model):
    """Create a :class:`Model` instance from an `SSDVGG16300Model` model.
    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first substract the first
        element of preprocessing from the input and then divide the input
        by the second element.
    """

    def __init__(
            self,
            bounds=(0, 255),
            channel_axis=3,
            preprocessing=(0, 1)):

        super(SSDVGG16300Model, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        self._task = 'det'
        self.model = hub.Module(name="ssd_vgg16_300_coco2017")

    def predictions(self, image):
        """Get prediction for input image
        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [h, w, c] ndarry format, BGR, 0 ~ 255
        Returns
        -------
        res (list[dict]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
            data (list): 检测结果，list的每一个元素为 dict，各字段为:
                confidence (float): 识别的置信度；
                label (str): 标签；
                left (int): 边界框的左上角x坐标；
                top (int): 边界框的左上角y坐标；
                right (int): 边界框的右下角x坐标；
                bottom (int): 边界框的右下角y坐标；
            save_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)。


        def object_detection(paths=None,
                             images=None,
                             batch_size=1,
                             use_gpu=False,
                             output_dir='detection_result',
                             score_thresh=0.5,
                             visualization=True)
        """

        predictions = self.model.object_detection(images=[image], score_thresh=0.2, visualization=False)
        return predictions

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task