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

"""Provide base classes that define what is an adversarial for object detection models."""

import math
from .base import Criterion
import numpy as np


class TargetClassMiss(Criterion):
    """ Defines adversarials as images for which the target class is not
    in the detection result.
    """

    def __init__(self, target_class):
        super(TargetClassMiss, self).__init__()
        self._target_class = target_class

    def target_class(self):
        """Return target class."""
        return self._target_class

    def name(self):
        """Return ctiterion name."""
        return 'TargetClassMiss'

    def is_adversarial(self, predictions, annotation):
        """Decides if predictions for an image are adversarial."""
        if predictions is None:
            return True
        return self._target_class not in predictions['classes']


class RegionalTargetClassMiss(Criterion):
    """Defines adversarials as images for which the target class in target region is not
    in the detection result.
    """

    def __init__(self, target_class, target_region):
        super(RegionalTargetClassMiss, self).__init__()
        self._target_class = target_class
        self._target_retion = np.array(target_region).astype(int)

    def target_class(self):
        """Return target class."""
        return self._target_class

    def target_region(self):
        """Return target region."""
        return self._target_retion

    def name(self):
        """Return ctiterion name."""
        return 'RegionalTargetClassMiss'

    def is_adversarial(self, predictions, annotation):
        """Decides if predictions for an image are adversarial."""
        if predictions is None:
            return True
        bbox_list = predictions['boxes']
        class_list = predictions['classes']
        for bbox_pred, cls_pred in zip(bbox_list, class_list):
            iou = self._get_IoU(bbox_pred, self._target_retion)
            if iou > 0 and cls_pred == self._target_class:
                return False
        return True

    @staticmethod
    def _get_IoU(bbox1, bbox2):
        bi = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
              min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])]
        ih = bi[2] - bi[0] + 1
        iw = bi[3] - bi[1] + 1
        if iw > 0 and ih > 0:
            # compute overlap (IoU) = area of intersection / area of union
            ua = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1) + \
                 (bbox2[2] - bbox2[0] + 1) * \
                (bbox2[3] - bbox2[1] + 1) - iw * ih
            ov = iw * ih / ua
            return ov
        else:
            return 0.0


class TargetClassMissGoogle(Criterion):
    """Defines adversarials as images for which the target class is not
    in the Google object detection result.
    """

    def __init__(self, target_class):
        super(TargetClassMissGoogle, self).__init__()
        self._target_class = target_class

    def target_class(self):
        """Return target class."""
        return self._target_class

    def name(self):
        """Return ctiterion name."""
        return '{}-{}'.format(
            self.__class__.__name__, self.target_class())

    def is_adversarial(self, predictions):
        """Decides if predictions for an image are adversarial."""
        if predictions is None:
            return True
        assert isinstance(predictions, list), 'Predictions should be list.'
        for pred in predictions:
            if pred['name'].lower() == self._target_class.lower():
                return False

        return True


class WeightedAP(Criterion):
    """Defines adversarials as weighted AP value
    larger than given threshold.
    """

    _defaults = {
        "alpha": 0.001,
        "lambda_tp_area": 0,
        "lambda_tp_dis": 0,
        "lambda_tp_cs": 0,
        "lambda_tp_cls": 1,
        "lambda_fp_area": 0.1,
        "lambda_fp_cs": 0,
        'lambda_fn_area': 0.1,
        'lambda_fn_cs': 0,
        'a_set': [1, 1, 1, 0.1],
        'MINOVERLAP': 0.5,
    }

    @classmethod
    def get_defaults(cls, n):
        """Return default value of n.
        Parameters
        ----------
        n : str
            Key of the defalut dictionary.
        """
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, Height, Width, distance_th, print_f=False):

        self.__dict__.update(self._defaults)  # set up default values
        self.Height = float(Height)
        self.Width = float(Width)
        self.th_is_adv = distance_th
        self.print_f = print_f
        self.a_tp = self.a_set[0]
        self.a_fp = self.a_set[1]
        self.a_fn = self.a_set[2]
        self.a_er = self.a_set[3]

    def name(self):
        """Return ctiterion name."""
        return 'WeightedAP'

    def is_adversarial(self, predictions, annotation):
        """Decides if predictions for an image are adversarial."""
        if predictions is None:
            return None
        return self.distance_score(annotation, predictions) > self.th_is_adv

    def _get_bb_area(self, bbox):
        return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)

    def _get_IoU(self, obj_1, obj_2):
        bb = obj_1["bbox"]
        bbgt = obj_2["bbox"]
        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
              min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
        iw = bi[2] - bi[0] + 1
        ih = bi[3] - bi[1] + 1
        if iw > 0 and ih > 0:
            # compute overlap (IoU) = area of intersection / area of union
            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                 (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
            ov = iw * ih / ua
            return ov
        else:
            return 0.0

    def _find_by_idx(self, idx, source_dic_list):
        for temp_obj in source_dic_list:
            if temp_obj['index'] == idx:
                return temp_obj
        return {}

    def _get_largest_bb_area(self, obj_list):
        temp_max = 1
        for temp_obj in obj_list:
            bb = temp_obj['bbox']
            bb_area = self._get_bb_area(bb)
            if bb_area > temp_max:
                temp_max = bb_area

        return temp_max

    def _get_total_bb_area(self, obj_list):
        total_area = 1
        for temp_obj in obj_list:
            bb = temp_obj['bbox']
            bb_area = self._get_bb_area(bb)
            total_area += bb_area
        return total_area

    def _get_largest_bb_edge(self, obj_list):
        temp_max = -1
        for temp_obj in obj_list:
            bb = temp_obj['bbox']
            if abs(bb[2] - bb[0]) > temp_max:
                temp_max = abs(bb[2] - bb[0])
            if abs(bb[3] - bb[1]) > temp_max:
                temp_max = abs(bb[3] - bb[1])
        return temp_max

    def _sort_by_conf(self, ori_list, source_dic_list):
        tup_list = []
        if len(ori_list) <= 1:
            return ori_list
        for temp in ori_list:
            temp_obj = self._find_by_idx(temp, source_dic_list)
            if not temp_obj:
                raise ValueError('object cannot be found by index.')
            tup_list.append((temp_obj['index'], temp_obj['confident_score']))
        tup_list.sort(key=lambda tup: tup[1])
        return [x[0] for x in tup_list]

    def _sort_match_dic(self, ori_index_dic, source_dic_list):
        sorted_dic = {}
        for temp_key in ori_index_dic.keys():
            temp_list = ori_index_dic[temp_key]
            if len(temp_list) <= 1:
                sorted_dic[temp_key] = temp_list
            else:
                sorted_dic[temp_key] = self._sort_by_conf(
                    temp_list, source_dic_list)
        return sorted_dic

    def _get_fn_list(self, tp_match_dic, source_list):
        dst_list = []
        for temp_source in source_list:
            flag_found = False
            for temp_idx_pair in tp_match_dic.keys():
                if (temp_source['index'] in tp_match_dic[temp_idx_pair]):
                    flag_found = True
            if not flag_found:
                dst_list.append(temp_source)
        return dst_list

    def _get_bb_distance(self, bb1, bb2):
        c1 = [0.5 * (bb1[2] + bb1[0]), 0.5 * (bb1[3] + bb1[1])]
        c2 = [0.5 * (bb2[2] + bb2[0]), 0.5 * (bb2[3] + bb2[1])]
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def distance_score(self, gt_dic, pd_dic):
        """Compute metric distance between given two detection results.
        Parameters
        ----------
        gt_dic : dict
            The ground truth annotation which contains: scores, boxes and classes.
        pd_dic : dict
            The target output form detector which contains: scores, boxes and classes.
        """
        gt_list = self._dic2list(gt_dic)
        pd_list = self._dic2list(pd_dic)
        return self._compute_score(gt_list, pd_list)

    def _dic2list(self, dic):
        res_list = []
        for idx, key in enumerate(dic.keys()):
            if idx == 0:
                for sub_idx in range(len(dic[key])):
                    res_list.append({'index': sub_idx})

            if key == 'scores':
                temp_name = 'confident_score'
            elif key == 'boxes':
                temp_name = 'bbox'
            elif key == 'classes':
                temp_name = 'class_name'
            else:
                raise ValueError('Invalid key.')
            for sub_idx, temp_obj in enumerate(dic[key]):
                if temp_name is 'bbox':
                    temp_obj = [
                        temp_obj[1],
                        temp_obj[0],
                        temp_obj[3],
                        temp_obj[2]]
                res_list[sub_idx][temp_name] = temp_obj
        return res_list

    def _compute_score(self, gt_obj_list, pd_obj_list):
        '''
        Notes
        -----
        compute metirc distance score for two results from object detection.
        input:
            pd_obj_list: object list of prediction
            gt_obj_list: object list of ground gruth
            obj = {
                'class_name' : 'car'
                'bbox' : '634 663 787 913' string of [left, up, right, down] splited by ' '
                'confident score' : 0.9918241
                'index' : 0
            }
        '''

        tp_match_dic = {}  # {pd_idx : [gt_idx1, gt_idx2...]}
        for pd_obj in pd_obj_list:
            tp_match_dic[pd_obj['index']] = []
            for gt_obj in gt_obj_list:
                IoU = self._get_IoU(pd_obj, gt_obj)
                # and gt_obj['class_name'] == pd_obj['class_name']:
                if IoU >= self.MINOVERLAP:
                    tp_match_dic[pd_obj['index']].append(gt_obj['index'])

        tp_match_dic = self._sort_match_dic(tp_match_dic, gt_obj_list)

        tp_pair = []
        fp_pd = []
        for temp_idx in tp_match_dic.keys():
            if not tp_match_dic[temp_idx]:
                fp_pd.append(self._find_by_idx(temp_idx, pd_obj_list))
            else:
                tp_pair.append(
                    (self._find_by_idx(
                        temp_idx, pd_obj_list), self._find_by_idx(
                        tp_match_dic[temp_idx][0], gt_obj_list)))

        fn_gt = self._get_fn_list(tp_match_dic, gt_obj_list)
        self.largest_area_gt = self._get_largest_bb_area(gt_obj_list)
        self.largest_edge_gt = self._get_largest_bb_edge(gt_obj_list)
        self.total_area_gt = self._get_total_bb_area(gt_obj_list)
        self.total_area_pd = self._get_total_bb_area(pd_obj_list)

        cum_tp_penal = 0.0
        for temp_tp_pair in tp_pair:
            results = self._tp_panelize(temp_tp_pair)
            distance = results['distance']
            area_dif = results['area_dif']
            cs_dif = results['cs_dif']
            class_dif = results['class_dif']
            temp_tp_penal = self.lambda_tp_dis * distance + self.lambda_tp_area * area_dif \
                + self.lambda_tp_cs * cs_dif + self.lambda_tp_cls * class_dif
            cum_tp_penal += temp_tp_penal
        if self.print_f:
            print('cum tp: ', cum_tp_penal)
        if len(tp_pair) > 1:
            cum_tp_penal /= len(tp_pair)

        cum_fp_penal = 0.0
        for temp_fp_pd in fp_pd:
            area, cs = self._fp_fn_panelize(temp_fp_pd)
            drop_func_out = self._factor_func(
                self.total_area_pd / (self.Height * self.Width))
            temp_fp_panel = self.lambda_fp_area * drop_func_out * \
                area / self.total_area_pd + self.lambda_fp_cs * cs
            cum_fp_penal += temp_fp_panel
        if self.print_f:
            print('cum fp: ', cum_fp_penal)
        if len(fp_pd) > 1:
            cum_fp_penal /= len(fp_pd)

        cum_fn_penal = 0.0
        for temp_fn_gt in fn_gt:
            area, cs = self._fp_fn_panelize(temp_fn_gt)
            drop_func_out = self._factor_func(
                self.total_area_gt / (self.Height * self.Width))
            temp_fn_panel = self.lambda_fn_area * drop_func_out * \
                area / self.total_area_gt + self.lambda_fn_cs * cs
            cum_fn_penal += temp_fn_panel
        if self.print_f:
            print('cum fn: ', cum_fn_penal)
        if len(fn_gt) > 1:
            cum_fn_penal /= len(fn_gt)

        if (len(tp_pair) + len(fp_pd) + len(fn_gt)) == 0:
            err_panel = 0
        else:
            err_panel = float((len(fp_pd) + len(fn_gt))) / \
                (len(tp_pair) + len(fp_pd) + len(fn_gt))
        if self.print_f:
            print('tp: ', len(tp_pair), ' cum_tp_penal: ', cum_tp_penal)
            print('fp: ', len(fp_pd), ' cum_fp_penal: ', cum_fp_penal)
            print('fn: ', len(fn_gt), ' cum_fn_penal: ', cum_fn_penal)
            print(
                'total num: ',
                len(tp_pair) +
                len(fp_pd) +
                len(fn_gt),
                ' err_panel: ',
                err_panel)

        score_final = (self.a_tp * cum_tp_penal + self.a_fp * cum_fp_penal + self.a_fn
                       * cum_fn_penal + self.a_er * err_panel) \
            / (self.a_tp + self.a_fp + self.a_fn + self.a_er)

        return score_final

    def _factor_func(self, x):

        x = float(x)
        if x != 0:
            return x / (x + self.alpha)
        return x

    def _tp_panelize(self, obj_pair):
        bb0 = obj_pair[0]['bbox']
        bb1 = obj_pair[1]['bbox']
        distance = self._get_bb_distance(bb0, bb1)
        area0 = self._get_bb_area(bb0)
        area1 = self._get_bb_area(bb1)
        area_dif = abs(area0 - area1)
        cs_dif = abs(
            float(
                obj_pair[0]['confident_score']) -
            float(
                obj_pair[1]['confident_score']))
        class_dif = 0
        if obj_pair[0]['class_name'] != obj_pair[1]['class_name']:
            class_dif = 1
        return {'distance': distance, 'area_dif': area_dif, 'cs_dif': cs_dif, 'class_dif': class_dif}

    def _fp_fn_panelize(self, obj):
        bb = obj['bbox']
        area = self._get_bb_area(bb)
        cs = float(obj['confident_score'])
        return area, cs
