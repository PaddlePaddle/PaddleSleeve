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
from attack.models.base import Model
from attack.utils.tools import denormalize_image
import os
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
import sys

import ppdet
from ppdet.core.workspace import create, load_config
from ppdet.utils.checkpoint import load_weight
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.utils.colormap import colormap
from PIL import Image, ImageOps, ImageFile, ImageDraw
import copy
from PIL import Image, ImageOps, ImageFile, ImageDraw
import matplotlib.pyplot as plt

__all__ = [
    'PPdet_Yolov3_Model',
    'PPdet_Rcnn_Model',
    'PPdet_Detr_Model',
    'PPdet_SSD_Model'
]

config_dir = os.path.dirname(os.path.realpath(__file__ + '../../..')) + '/configs'


def _draw_bbox(image, bboxes, ignore_thresh=0.1, mask_cls=[]):
    """
        Draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        catid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if score < ignore_thresh or catid in mask_cls:
            continue

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        xmin, ymin, xmax, ymax = bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill=color)
        # draw label
        text = "{} {:.2f}".format(catid, score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image


def _de_sigmoid(x, eps=1e-7):
    """
        decode sigmoid
    """
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


class PPdet_Model(Model):
    """
    Base wrapper class for ppdet models trained on coco dataset
    """

    def __init__(
            self,
            bounds=(0, 1),
            channel_axis=0,
            preprocessing=(0, 1)):
        super(PPdet_Model, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

    def predictions_tensor(self, image_tensor):
        """Get prediction for input image
        Parameters
        ----------
        image_tensor : `paddle.Tensor`
            The input image in [c, h, w] ndarry format, BGR, 0 ~ 1
        Returns
        -------
        predictions : `paddle.Tensor`
            The prediction result as a 2D tensor

        def object_detection(paths=None,
                             images=None,
                             batch_size=1,
                             use_gpu=False,
                             output_dir='detection_result',
                             score_thresh=0.5,
                             visualization=True)
        """

        if len(paddle.shape(image_tensor)) < 4:
            image_tensor = paddle.unsqueeze(image_tensor, axis=0)
        assert self._data['image'].shape == image_tensor.shape

        data1 = copy.deepcopy(self._data)
        data1['image'] = image_tensor
        self._model.eval()
        outs1 = self._model(data1)
        return outs1

    def predictions(self, image):
        """Get prediction for input image
        Parameters
        ----------
        image : `numpy.ndarray`
            The input image in [c, h, w] ndarry format, RGB, 0 ~ 1
        Returns
        -------
        bbox : `numpy.ndarray`
            The prediction result with shape [N, B, 6], where N is the batch size, B is the number of bounding boxes. The 6 attributes
            of each bounding box are class, confidence, and its position respectively
        """

        self._model.eval()
        image = self._preprocessing(image)
        image_tensor = paddle.to_tensor(image, dtype='float32', stop_gradient=True)
        predictions_tensor = self.predictions_tensor(image_tensor)
        return predictions_tensor['bbox'].numpy()

    def model_task(self):
        """Get the task that the model is used for."""
        return self._task

    def load_image(self, image):
        images = [os.path.join(os.path.dirname(__file__), '../../../', single_image) for single_image in image]
        self._dataset.set_images(images*1)
        loader1 = create('TestReader')(self._dataset, 0)
        for i, data in enumerate(loader1):
            self._data = data
            self._scale_factor = data['scale_factor']
        return denormalize_image(paddle.squeeze(self._data['image'][0]), self.MEAN, self.STD)

    def load_weight(self, weight):
        load_weight(self._model, weight)

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def parameters(self):
        return self._model.parameters()

    def loss(self, data):
        self._model.train()
        return self._model(data)

    def _gather_feats(self, image):
        raise NotImplementedError

    def _adv_loss(self, image):
        raise NotImplementedError


class PPdet_Yolov3_Model(PPdet_Model):
    """
    Base wrapper class for ppdet yolov3 models trained on coco dataset

    Configs:
            backbone: varies
            neck: YOLOv3FPN
            yolo_head: YOLOv3Head
            post_process: BBoxPostProcess
    """

    def __init__(
            self,
            name,
            bounds=(0, 1),
            channel_axis=0,
            preprocessing=(0, 1),
            pretrained=True):
        super(PPdet_Yolov3_Model, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        cfg_file = "{0}/yolov3/{1}.yml".format(config_dir, name)
        self._cfg = load_config(cfg_file)
        self._cfg['norm_type'] = 'bn'
        self.architecture = 'yolov3'
        self._model = create(self._cfg.architecture)
        self._model.load_meanstd(self._cfg['TestReader']['sample_transforms'])
        self.MEAN = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['mean']
        self.STD = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['std']
        self._preprocessing = paddle.vision.transforms.Normalize(mean=self.MEAN, std=self.STD)
        self._scale_factor = None
        self._num_classes = self._cfg['num_classes']

        if pretrained:
            #weight_path = "ppdet://models/{}.pdparams".format(name)
            if self._cfg.get('base_model_name', False):
                weight_path = "ppdet://models/{}.pdparams".format(self._cfg['base_model_name'])
            else:
                weight_path = "ppdet://models/{}.pdparams".format(name)
            load_weight(self._model, weight_path)

        assert isinstance(self._model, paddle.nn.Layer), 'Unrecognized model'

        self._model.eval()

        self._task = 'det'
        self._data = None
        self._dataset = self._cfg['TestDataset']

    def _gather_feats(self, image):
        if len(paddle.shape(image)) < 4:
            image = paddle.unsqueeze(image, axis=0)
        assert self._data['image'].shape == image.shape

        data1 = copy.deepcopy(self._data)
        data1['image'] = image

        self._model.eval()

        # start forward inference
        body_feats = self._model.backbone(data1)
        neck_feats = self._model.neck(body_feats, False)

        assert len(neck_feats) == len(self._model.yolo_head.anchors)
        yolo_outputs = []
        for i, feat in enumerate(neck_feats):
            yolo_output = self._model.yolo_head.yolo_outputs[i](feat)
            if self._model.yolo_head.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

        decode = self._model.post_process.decode
        if self._model.yolo_head.iou_aware:
            y = []
            for i, out in enumerate(yolo_outputs):
                na = len(self._model.yolo_head.anchors[i])
                ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                b, c, h, w = x.shape
                no = c // na
                x = x.reshape((b, na, no, h * w))
                ioup = ioup.reshape((b, na, 1, h * w))
                obj = x[:, :, 4:5, :]
                ioup = F.sigmoid(ioup)
                obj = F.sigmoid(obj)
                obj_t = (obj ** (1 - self._model.yolo_head.iou_aware_factor)) * (
                        ioup ** self._model.yolo_head.iou_aware_factor)
                obj_t = _de_sigmoid(obj_t)
                loc_t = x[:, :, :4, :]
                cls_t = x[:, :, 5:, :]
                y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                y_t = y_t.reshape((b, c, h, w))
                y.append(y_t)
            yolo_outputs = y
        bboxes, scores = decode(yolo_outputs, self._model.yolo_head.mask_anchors,
                        data1['im_shape'], data1['scale_factor'])
        bbox_pred, bbox_num, _ = self._model.post_process.nms(bboxes, scores, self._model.post_process.num_classes)

        return {'body_feats':body_feats, 'neck_feats':neck_feats, 'head_feats':yolo_outputs,
                'bboxes':bboxes, 'scores':scores, 'bbox_pred':bbox_pred, 'bbox_num':bbox_num}

    def adv_loss(self, features, target_class, confidence=0.1):
        head_feats = features['head_feats']
        scores_logit = []
        probs_logit = []
        for feat in head_feats:
            N, _, X, Y = feat.shape
            boxes = paddle.transpose(feat, [0, 2, 3, 1])
            boxes = paddle.reshape(boxes, [N, X, Y, 3, 5+self._num_classes])
            score_logit = boxes[0, :, :, :, 4:5]
            prob_logit = boxes[0, :, :, :, 5:]
            score_logit = paddle.reshape(score_logit, [-1, 1])
            prob_logit = paddle.reshape(prob_logit, [-1, self._num_classes])
            scores_logit.append(score_logit)
            probs_logit.append(prob_logit)
        box_scores_logit = paddle.concat(scores_logit, axis=0)
        cls_prob_logits = paddle.concat(probs_logit, axis=0)
        box_scores = paddle.nn.Sigmoid()(box_scores_logit)
        cls_probs = paddle.nn.Sigmoid()(cls_prob_logits)
        box_scores = box_scores * cls_probs
        target_scores = box_scores[:, target_class]
        boi = target_scores > confidence
        # logits = paddle.masked_select(cls_prob_logits[:, self._target_class], boi)
        # loss = paddle.mean(logits)
        boi = paddle.unsqueeze(boi, axis=1)
        logits = cls_prob_logits - paddle.min(cls_prob_logits, axis=1, keepdim=True)
        logits = logits * boi
        nonzero_idx = paddle.tolist(paddle.squeeze(paddle.nonzero(logits[:, target_class])))
        if nonzero_idx == []:
            return paddle.zeros([1])
        logits = logits[nonzero_idx]
        target_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(target_class, dtype='int32'), self._num_classes)
        loss = paddle.max(logits * target_onehot, axis=1) - paddle.max(logits * (1 - target_onehot), axis=1)
        return loss


class PPdet_Rcnn_Model(PPdet_Model):
    """
    Base wrapper class for ppdet rcnn models trained on coco dataset

    Configs:
            backbone: resnet
            neck: None
            rpn_head: RPNHead
            bbox_head: BBoxHead
            post_process: BBoxPostProcess
    """

    def __init__(
            self,
            name,
            bounds=(0, 1),
            channel_axis=0,
            preprocessing=(0, 1),
            pretrained=True,
            cascade=False):
        super(PPdet_Rcnn_Model, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)
        self._cascade = cascade

        if self._cascade:
            cfg_file = "{0}/cascade_rcnn/{1}.yml".format(config_dir, name)
        else:
            cfg_file = "{0}/faster_rcnn/{1}.yml".format(config_dir, name)

        self._cfg = load_config(cfg_file)
        self._cfg['norm_type'] = 'bn'
        self.architecture = 'rcnn'
        self._model = create(self._cfg.architecture)
        self._model.load_meanstd(self._cfg['TestReader']['sample_transforms'])
        self.MEAN = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['mean']
        self.STD = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['std']
        self._preprocessing = paddle.vision.transforms.Normalize(mean=self.MEAN, std=self.STD)
        self._scale_factor = None

        if pretrained:
            #weight_path = "ppdet://models/{}.pdparams".format(name)
            if self._cfg.get('base_model_name', False):
                weight_path = "ppdet://models/{}.pdparams".format(self._cfg['base_model_name'])
            else:
                weight_path = "ppdet://models/{}.pdparams".format(name)
            load_weight(self._model, weight_path)

        assert isinstance(self._model, paddle.nn.Layer), 'Unrecognized model'

        self._model.eval()

        self._task = 'det'
        self._data = None
        self._dataset = self._cfg['TestDataset']

    def _gather_feats(self, image):
        if len(paddle.shape(image)) < 4:
            image = paddle.unsqueeze(image, axis=0)
        assert self._data['image'].shape == image.shape

        data1 = copy.deepcopy(self._data)
        data1['image'] = image

        self._model.eval()

        # start forward inference
        body_feats = self._model.backbone(data1)
        if self._model.neck is not None:
            body_feats = self._model.neck(body_feats)

        # print(body_feats)
        rois, rois_num, _ = self._model.rpn_head(body_feats, data1)
        # ----------------------------------------------------------------
        self._model.bbox_head
        if self._cascade:
            # Cascade Head forward
            pred_bbox = None
            head_out_list = []
            for i in range(self._model.bbox_head.num_cascade_stages):
                if i > 0:
                    rois, rois_num = self._model.bbox_head._get_rois_from_boxes(pred_bbox,
                                                                    data1['im_shape'])

                rois_feat = self._model.bbox_head.roi_extractor(body_feats, rois, rois_num)
                bbox_feat = self._model.bbox_head.head(rois_feat, i)
                scores = self._model.bbox_head.bbox_score_list[i](bbox_feat)
                deltas = self._model.bbox_head.bbox_delta_list[i](bbox_feat)
                head_out_list.append([scores, deltas, rois])
                pred_bbox = self._model.bbox_head._get_pred_bbox(deltas, rois, self._model.bbox_head.bbox_weight[i])

            scores, deltas, refined_rois = self._model.bbox_head.get_prediction(
                head_out_list)
            logits_list = [head[0] for head in head_out_list]
            cls_prob_logits = paddle.add_n(logits_list) / self._model.bbox_head.num_cascade_stages
            preds = (deltas, scores)
        else:
            # bbox head forward
            rois_feat = self._model.bbox_head.roi_extractor(body_feats, rois, rois_num)
            bbox_feat = self._model.bbox_head.head(rois_feat)
            if self._model.bbox_head.with_pool:
                feat = paddle.nn.functional.adaptive_avg_pool2d(bbox_feat, output_size=1)
                feat = paddle.squeeze(feat, axis=[2, 3])
            else:
                feat = bbox_feat
            cls_prob_logits = self._model.bbox_head.bbox_score(feat)
            deltas = self._model.bbox_head.bbox_delta(feat)
            preds = self._model.bbox_head.get_prediction(cls_prob_logits, deltas)
            refined_rois = rois
        # ----------------------------------------------------------------

        im_shape = data1['im_shape']
        scale_factor = data1['scale_factor']
        bbox, bbox_num = self._model.bbox_post_process(preds, (refined_rois, rois_num),
                                                im_shape, scale_factor)

        # rescale the prediction back to origin image
        #bbox_pred = self._model.bbox_post_process.get_pred(bbox, bbox_num,
        #                                            im_shape, scale_factor)
        bboxes, bbox_pred, bbox_num = self._model.bbox_post_process.get_pred(bbox, bbox_num, im_shape, scale_factor)
        return {'body_feats': body_feats, 'preds': preds, 'cls_prob_logits': cls_prob_logits,
                'bbox_pred': bbox_pred, 'bbox_num': bbox_num}

    def adv_loss(self, features, target_class, confidence=0.1):
        cls_prob_logits = features['cls_prob_logits']
        scores = features['preds'][1]
        target_scores = scores[:, target_class]
        boi = target_scores > confidence
        # logits = paddle.masked_select(cls_prob_logits[:, self._target_class], boi)
        # loss = paddle.mean(logits)

        boi = paddle.unsqueeze(boi, axis=1)
        logits = cls_prob_logits - paddle.min(cls_prob_logits, axis=1, keepdim=True)
        logits = logits * boi
        if logits.shape[-1] != self._cfg['num_classes']:
            logits = logits[:, :-1]
        nonzero_idx = paddle.tolist(paddle.squeeze(paddle.nonzero(logits[:, target_class])))
        if nonzero_idx == []:
            return paddle.zeros([1])
        logits = logits[nonzero_idx]
        target_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(target_class, dtype='int32'), self._cfg['num_classes'])
        loss = paddle.max(logits * target_onehot, axis=1) - paddle.max(logits * (1 - target_onehot), axis=1)
        return loss

class PPdet_Detr_Model(PPdet_Model):
    """
    Base wrapper class for ppdet detr models trained on coco dataset

    Configs:
            backbone: resnet50
            transformer: DETRTransformer
            detr_head: DETRHead
            post_process: DETRBBoxPostProcess

    """

    def __init__(
            self,
            name,
            bounds=(0, 1),
            channel_axis=0,
            preprocessing=(0, 1),
            pretrained=True):
        super(PPdet_Detr_Model, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        cfg_file = "{0}/detr/{1}.yml".format(config_dir, name)
        self._cfg = load_config(cfg_file)
        self._cfg['norm_type'] = 'bn'
        self.architecture = 'detr'
        self._model = create(self._cfg.architecture)
        self._model.load_meanstd(self._cfg['TestReader']['sample_transforms'])
        self.MEAN = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['mean']
        self.STD = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['std']
        self._preprocessing = paddle.vision.transforms.Normalize(mean=self.MEAN, std=self.STD)
        self._scale_factor = None

        if pretrained:
            #weight_path = "ppdet://models/{}.pdparams".format(name)
            if self._cfg.get('base_model_name', False):
                weight_path = "ppdet://models/{}.pdparams".format(self._cfg['base_model_name'])
            else:
                weight_path = "ppdet://models/{}.pdparams".format(name)
            load_weight(self._model, weight_path)

        assert isinstance(self._model, paddle.nn.Layer), 'Unrecognized model'

        self._model.eval()

        self._task = 'det'
        self._data = None
        self._dataset = self._cfg['TestDataset']

    def load_image(self, image):
        # img_path = os.path.join(os.path.dirname(__file__), '../../../', image)
        # images = [img_path]
        images = [os.path.join(os.path.dirname(__file__), '../../../', single_image) for single_image in image]
        self._cfg['TestReader']['batch_transforms'][0]['PadMaskBatch']['return_pad_mask'] = True
        self._dataset.set_images(images*1)
        loader1 = create('TestReader')(self._dataset, 0)
        for i, data in enumerate(loader1):
            self._data = data
            self._scale_factor = data['scale_factor']
        return denormalize_image(paddle.squeeze(self._data['image']), self.MEAN, self.STD)

    def _gather_feats(self, image):
        if len(paddle.shape(image)) < 4:
            image = paddle.unsqueeze(image, axis=0)
        assert self._data['image'].shape == image.shape

        data1 = copy.deepcopy(self._data)
        data1['image'] = image

        self._model.eval()

        # start forward inference
        # Backbone
        body_feats = self._model.backbone(data1)

        # Transformer
        out_transformer = self._model.transformer(body_feats, data1['pad_mask'])

        # DETR Head
        preds = self._model.detr_head(out_transformer, body_feats)
        bbox, bbox_num = self._model.post_process(preds, data1['im_shape'],
                                               data1['scale_factor'])
        return {'body_feats': body_feats, 'transformer_out': out_transformer, 'preds': preds,
                'cls_prob_logits': preds[1], 'bbox_pred': bbox, 'bbox_num': bbox_num}

    def adv_loss(self, features, target_class, confidence=0.1):
        cls_prob_logits = paddle.squeeze(features['cls_prob_logits'])
        scores = paddle.squeeze(paddle.nn.functional.softmax(cls_prob_logits))
        # print(scores)
        target_scores = scores[:, target_class]
        boi = target_scores > confidence
        # logits = paddle.masked_select(cls_prob_logits[:, self._target_class], boi)
        # loss = paddle.mean(logits)

        boi = paddle.unsqueeze(boi, axis=1)
        logits = cls_prob_logits - paddle.min(cls_prob_logits, axis=1, keepdim=True)
        logits = logits * boi
        if logits.shape[-1] != self._cfg['num_classes']:
            logits = logits[:, :-1]
        nonzero_idx = paddle.tolist(paddle.squeeze(paddle.nonzero(logits[:, target_class])))
        if nonzero_idx == []:
            return paddle.zeros([1])
        logits = logits[nonzero_idx]
        target_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(target_class, dtype='int32'), self._cfg['num_classes'])
        loss = paddle.max(logits * target_onehot, axis=1) - paddle.max(logits * (1 - target_onehot), axis=1)
        return loss


class PPdet_SSD_Model(PPdet_Model):
    """
    Base wrapper class for ppdet ssd models trained on coco dataset

    Configs:
        backbone (nn.Layer): backbone instance
        ssd_head (nn.Layer): `SSDHead` instance
        post_process (object): `BBoxPostProcess` instance
    """
    def __init__(
            self,
            name,
            bounds=(0, 1),
            channel_axis=0,
            preprocessing=(0, 1),
            pretrained=True):
        super(PPdet_SSD_Model, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis,
            preprocessing=preprocessing)

        cfg_file = "{0}/ssd/{1}.yml".format(config_dir, name)
        self._cfg = load_config(cfg_file)
        self._cfg['norm_type'] = 'bn'
        self.architecture = 'ssd'
        self._model = create(self._cfg.architecture)
        self._model.load_meanstd(self._cfg['TestReader']['sample_transforms'])
        self.MEAN = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['mean']
        self.STD = self._cfg['TestReader']['sample_transforms'][2]['NormalizeImage']['std']
        self._preprocessing = paddle.vision.transforms.Normalize(mean=self.MEAN, std=self.STD)
        self._scale_factor = None
        self._num_classes = self._cfg['num_classes']

        if pretrained:
            weight_path = "ppdet://models/{}.pdparams".format(name)
            load_weight(self._model, weight_path)

        assert isinstance(self._model, paddle.nn.Layer), 'Unrecognized model'

        self._model.eval()

        self._task = 'det'
        self._data = None
        self._dataset = self._cfg['TestDataset']

    def load_image(self, image):
        images = [os.path.join(os.path.dirname(__file__), '../../../', single_image) for single_image in image]
        self._dataset.set_images(images*1)
        loader1 = create('TestReader')(self._dataset, 0)
        for i, data in enumerate(loader1):
            self._data = data
            self._scale_factor = data['scale_factor']
        return denormalize_image(paddle.squeeze(self._data['image'][0]), self.MEAN, self.STD)

    def _gather_feats(self, image):
        if len(paddle.shape(image)) < 4:
            image = paddle.unsqueeze(image, axis=0)
        assert self._data['image'].shape == image.shape

        data1 = copy.deepcopy(self._data)
        data1['image'] = image

        self._model.eval()

        body_feats = self._model.backbone(data1)
        # SSD Head
        preds, anchors = self._model.ssd_head(body_feats, image)

        # SSD post process
        bbox, bbox_num = self._model.post_process(preds, anchors, data1['im_shape'], data1['scale_factor'])
        return {'body_feats': body_feats, 'preds': preds, 'anchors': anchors, 'box_preds': preds[0],
                'cls_scores': preds[1], 'bbox_pred': bbox, 'bbox_num': bbox_num}

    def adv_loss(self, features, target_class, confidence=0.1):
        """

        :param features:
        :param target_class:
        :param confidence:
        :return:
        """
        # print("box_preds:", features['box_preds'], "type(features['box_preds']):", type(features['box_preds']))
        # print("cls_scores:", features['cls_scores'], "type(features['cls_scores']):", type(features['cls_scores']))

        cls_prob_logits = paddle.squeeze(paddle.concat(features['cls_scores'], axis=1))
        scores = paddle.squeeze(paddle.nn.functional.softmax(cls_prob_logits))

        target_scores = scores[:, target_class]
        boi = target_scores > confidence

        boi = paddle.unsqueeze(boi, axis=1)
        logits = cls_prob_logits - paddle.min(cls_prob_logits, axis=1, keepdim=True)
        logits = logits * boi
        if logits.shape[-1] != self._cfg['num_classes']:
            logits = logits[:, :-1]
        nonzero_idx = paddle.tolist(paddle.squeeze(paddle.nonzero(logits[:, target_class])))
        if nonzero_idx == []:
            return paddle.zeros([1])
        logits = logits[nonzero_idx]
        target_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(target_class, dtype='int32'), self._cfg['num_classes'])
        loss = paddle.max(logits * target_onehot, axis=1) - paddle.max(logits * (1 - target_onehot), axis=1)
        return loss
