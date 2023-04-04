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

"""
The adaptive patch attack implementation.
Contains:
* adaptive patch position and size determination.
* adaptive patch apply for all images.
* Generate adversarial patch.

Author: tianweijuan
"""

from __future__ import absolute_import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))
import warnings
warnings.filterwarnings('ignore')
import glob
import cv2
import math
from PIL import Image

import xmltodict
import numpy

import argparse
import os
import sys
from pathlib import Path

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
from ppdet.core.workspace import create
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.data.source.category import get_categories
from ppdet.metrics import get_infer_results
from ppdet.utils.logger import setup_logger

from depreprocess.operator_composer import OperatorCompose
from load_data import *



import xmltodict
import math
import random
from past.utils import old_div




logger = setup_logger('train')

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for images to save on.")
    parser.add_argument(
        "--output_patch_dir",
        type=str,
        default="output_patch",
        help="Directory for images to save on.")

    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--target_img",
        type=str,
        default=None,
        help="Image path, infer image with masked on.")
    
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.6,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_txt",
        type=bool,
        default=False,
        help="Whether to save inference result in txt.")
    args = parser.parse_args()
    return args


batch_size = 40 #23
epochs = 60

class NPSCalculator(nn.Layer):
    """NMSCalculator: calculates the non-printability score of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = self.get_printability_array(printability_file, patch_side)
    
    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = paddle.sum(color_dist, 1)+0.000001
        color_dist = paddle.sqrt(color_dist)
        # only work with the min distance
        # change prod for min (find distance to closest color)
        color_dist_prod = paddle.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = paddle.sum(color_dist_prod,0)
        nps_score = paddle.sum(nps_score,0)
        return nps_score/paddle.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []
        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side[0], side[1]), red))
            printability_imgs.append(np.full((side[0], side[1]), green))
            printability_imgs.append(np.full((side[0], side[1]), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = paddle.to_tensor(printability_array)
        return pa

class TotalVariation(nn.Layer):
    """TotalVariation: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = paddle.sum(paddle.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = paddle.sum(paddle.sum(tvcomp1,0),0)
        tvcomp2 = paddle.sum(paddle.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = paddle.sum(paddle.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/paddle.numel(adv_patch)



def get_pcls(model, neck_feats):
    """
    Get pcls with given neck_feats.
    Args:
        model: paddle.model. PaddleDetection model.
        neck_feats: paddle.tensor. Inferenced result from detector.head.

    Returns:
        paddle.tensor. pcls tensor.
    """

    pcls_list = []
    conf_list = []
    
    for i, feat in enumerate(neck_feats):
        yolo_output = model.yolo_head.yolo_outputs[i](feat)
        
        if model.data_format == 'NHWC':
            yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
        p = yolo_output
        number_anchor = 3
        b, c, h, w = p.shape

        p = p.reshape((b, number_anchor, -1, h, w)).transpose((0, 1, 3, 4, 2))
        conf, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        pcls_list.append(pcls)
        conf_list.append(conf)
    return pcls_list, conf_list

def HA(model, feat_org, feat):
    vecorg_list = []
    vec_list = []
    
    for i in range(len(feat_org)):
        yolo_featorg = model.yolo_head.yolo_outputs[i](feat_org[i])
        yolo_feat = model.yolo_head.yolo_outputs[i](feat[i])
        vec_org = F.adaptive_avg_pool2d(yolo_featorg, (1,1)).squeeze(2).squeeze(2)
        vec = F.adaptive_avg_pool2d(yolo_feat, (1,1)).squeeze(2).squeeze(2)
        vecorg_list.append(vec_org)
        vec_list.append(vec)
    vecorg_list = paddle.to_tensor(vecorg_list)
    vec_list = paddle.to_tensor(vec_list)
    vecorg_max = paddle.fluid.layers.reduce_max(vecorg_list, 0)
    vec_max = paddle.fluid.layers.reduce_max(vec_list, 0)
    
    loss_ha = 1. / nn.MSELoss()(vecorg_max, vec_max)
    
    return loss_ha

def get_mask_coordination(_object):
    """
    Place mask coordination in variables.
    Args:
    maskfilename: Path for the xml file containing mask coordination.
    **kwargs: Other named arguments.
    """

    xmin = int(_object['bndbox']['xmin'])
    ymin = int(_object['bndbox']['ymin'])
    xmax = int(_object['bndbox']['xmax'])
    ymax = int(_object['bndbox']['ymax'])

    return xmin,ymin,xmax,ymax

class AttackNet(nn.Layer):
    """
    The attack_net implementation based on PaddlePaddle.
    As mentioned in the original paper, author proposes a novel expectation over transformation
    method that automatically learns the adversarial patch foucs on the original input image.
    The model aims to remain adversarial under mage transformations that occur in the real world.
    The original article refers to
    Athalye, A, et, al. "Synthesizing Robust Adversarial Examples."
    (http://proceedings.mlr.press/v80/athalye18b/athalye18b.pdf).
    Args:
        cfg (dict): The model definition to be attacked.
        dic (dict): The added patch size and object label definition.
    """

    def __init__(self, cfg, dic, h, w):
        super(AttackNet, self).__init__()
        
        self.trainer = Trainer(cfg, mode='test')
        self.trainer.load_weights(cfg.weights)
        self.label_id = int(dic['annotation']['label']['id'])
        
        mask_list = dic['annotation']['object']
        box_list = dic['annotation']['size']
        obj_list = dic['annotation']['obj_size']
          
        
        self.widtht, self.heightt = int(box_list['width']), int(box_list['height'])

        self.obj_xmin, self.obj_ymin, self.obj_w, self.obj_h = float(obj_list['xmin']), float(obj_list['ymin']), float(obj_list['width']), float(obj_list['height'])
        self.xmin, self.ymin, self.xmax, self.ymax = get_mask_coordination(mask_list)
        self.h, self.w = h, w 
        self.patch_scale_w, self.patch_scale_h = (self.xmax -self.xmin) / self.obj_w, (self.ymax- self.ymin) / self.obj_h
        self.patch_scale_xmin, self.patch_scale_ymin  = (self.xmin - self.obj_xmin)/self.obj_w, (self.ymin - self.obj_ymin) / self.obj_h
      
        self.init_inter_mask = paddle.fluid.initializer.Normal(loc=2.5, scale=0.8)         
        self.masked_inter = paddle.fluid.layers.create_parameter([1, 3,  self.ymax - self.ymin, self.xmax - self.xmin], 'float32', name="masked_inter", default_initializer=self.init_inter_mask)
        
        self.nnSigmoid = paddle.nn.Sigmoid()
        self.nnSoftmax = paddle.nn.Softmax()
        self.total_variation = TotalVariation()
        self.patch_transformer = PatchTransformer()
        self.patch_applier = PatchApplier()
        self.nps_calculator = NPSCalculator("./30values.txt", [self.ymax - self.ymin, self.xmax - self.xmin])
       
        self.mse = nn.MSELoss()

    def forward(self, input1, epoch):
                
        masked_inter_batch_val = paddle.clip(self.masked_inter, min=0., max=5.)
        masked_inter_batches_val = paddle.tanh(masked_inter_batch_val)
        orig_img = input1['image_org'] # need to modify the file in ppdet
        orig_img = paddle.cast(orig_img, dtype='float32')
        scale = 1.0 / 255.0
        orig_img *= scale  # (0, 1)
        label = input1["label"]
        path = input1["im_file"]
        
        adv_batch_t0, masked_inter_batches_val0 = self.patch_transformer(masked_inter_batches_val, label, (self.heightt, self.widtht), (self.patch_scale_w, self.patch_scale_h), (self.patch_scale_xmin, self.patch_scale_ymin), do_rotate=False, rand_loc=False)
        

        orig_img = orig_img.transpose((0, 3, 1, 2))
        
        p_img_batch = self.patch_applier(orig_img, adv_batch_t0)
               
        X_batch_re  = F.interpolate(p_img_batch, (self.h, self.w), mode='bilinear') 
        b, c, _, _ = X_batch_re.shape

        # normalization
        mean = paddle.to_tensor(np.array([0.485, 0.456, 0.406])[np.newaxis, :, np.newaxis, np.newaxis].repeat(b,axis=0))     
        std = paddle.to_tensor(np.array([0.229, 0.224, 0.225])[np.newaxis, :, np.newaxis, np.newaxis].repeat(b,axis=0))
        
        X_batch_re -= mean
        X_batch_re /= std

        self.trainer.model.eval()
        
        body_feats = self.trainer.model.backbone(input1)
        outs1 = self.trainer.model.neck(body_feats, False)
        
        input1['image'] = X_batch_re
        body_feats = self.trainer.model.backbone(input1)
        outs2 = self.trainer.model.neck(body_feats, False)
        
        loss_ha = HA(self.trainer.model, outs1, outs2)
        
        #print("loss_ha=========", loss_ha * 1e-3) # 1e-3
        pcls_list, conf_list = get_pcls(self.trainer.model, outs2)
          
        # tv variation
        tv_loss = self.total_variation(masked_inter_batches_val[0])
       
        # nps computation
        nps0 = self.nps_calculator(masked_inter_batches_val[0])
        #print("nps0=======", nps0*200, tv_loss)
        
        nps_loss = nps0 * 200                
        C_target = 0.
        
        for pcls, conf in zip(pcls_list, conf_list):
            b, anc, h, w, cls = pcls.shape    
            obj = F.sigmoid(conf)
            pcls_cls = F.sigmoid(pcls) 
            
            pcls_obj = paddle.reshape(pcls_cls[:, :, :, :, self.label_id], [b, anc*h*w])
            pcls_car = paddle.reshape(pcls_cls[:, :, :, :, 3], [b, anc*h*w]) # car with label 3 is the class we don't want to be attacked successfully
            pcls_obj = paddle.fluid.layers.reduce_max(pcls_obj, 1) 
            pcls_obj = paddle.fluid.layers.reduce_sum(pcls_obj, 0)

            pcls_car = paddle.fluid.layers.reduce_max(pcls_car, 1)
            pcls_car = paddle.fluid.layers.reduce_sum(pcls_car, 0)
            
            C_target += 0.8* pcls_obj
            C_target += 0.2* pcls_car


        #print(C_target/len(p_img_batch), nps_loss, loss_ha * 1e-3)
        loss = C_target/len(p_img_batch) + nps_loss + loss_ha * 1e-3 #+ tv_loss 
        pred = self.trainer.model(input1)
        
        return loss, pred, p_img_batch, [masked_inter_batches_val0]
        


def run(FLAGS, cfg):
    """
    construct input data and call the AttackNet to achieve the adversarial patch learning.
    Args:
        FLAGS(dict): configure parameters
        cfg(str): attacked model configs
    """

    # init depre
    f = open('patch_def/patch_def.xml')
    dic = xmltodict.parse(f.read())
    size = dic['annotation']['size']
    depre_settings = {'ImPermute': {},
                      'DenormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                           'input_channel_axis': 2, 'is_scale': True},
                      'Resize': {'target_size': (int(size['height']), int(size['width']), int(size['depth'])), 'keep_ratio': False, 'interp': 2}, #638, 850, 864, 1152
                      'Encode': {}
                      }
    depreprocessor = OperatorCompose(depre_settings)
    draw_threshold = FLAGS.draw_threshold
    
    loader, datainfo0  = _image2outs(FLAGS.infer_dir, FLAGS.infer_img, cfg)
    
    model_attack = AttackNet(cfg, dic, 320, 320)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.01, T_max=60, verbose=True)
    
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters = model_attack.parameters())
    steps_per_epoch = 1150 // batch_size 
    for epoch in range (epochs): 
        Obj_Num = 0
        loss_epoch = 0.
        for step_id, data in enumerate(loader):
            
            loss, outs_adv, data_adv, masked = model_attack(data, epoch)
            loss.backward() #retain_graph=True)
            opt.minimize(loss)
            print('step_id:', step_id, '======loss:', loss.numpy())
            loss_epoch += loss
            obj_num = ext_score(outs_adv, data, datainfo0, int(dic['annotation']['label']['id']))
            Obj_Num += obj_num
           
            print('Obj_Num=======:', Obj_Num, step_id, data_adv.shape)
            if epoch == epochs-1: #or Obj_Num < 600:
                for i in range(data_adv.shape[0]):
                    in_adv1 = data_adv[i].transpose((1, 2, 0)).detach().cpu().numpy()
                    in_adv1 = np.ascontiguousarray(in_adv1) * 255. 
                    in_adv1 = cv2.cvtColor(in_adv1, cv2.COLOR_RGB2BGR)
                    if not os.path.exists(FLAGS.output_dir):
                        os.makedirs(FLAGS.output_dir)
                
                    cv2.imwrite(FLAGS.output_dir + "/" + data["im_file"][i].split("/")[-1], in_adv1)
        
        print('epoch:', epoch, '======loss_epoch:', loss_epoch.numpy())#/23)  
        #scheduler.step()
        #if Obj_Num < 540:
        #    break

    if not os.path.exists(FLAGS.output_patch_dir):
        os.makedirs(FLAGS.output_patch_dir)
    
    for i in range(len(masked)):
        masked1 = masked[i][0, :, :, :].transpose((1, 2, 0)).detach().cpu().numpy()
        masked1 = np.ascontiguousarray(masked1) * 255.
        masked1 = cv2.cvtColor(masked1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.output_patch_dir+ "/adv_patch"+ str(i)+".png", masked1)


def _image2outs(infer_dir, infer_img, cfg):
    """
    construct the single input data for the post data process.
    Args:
        infer_dir(str): input data path
        infer_img(str): input data filename
        cfg(dict): attacked model definition
    """
    
    mode = 'test'
    dataset = cfg['{}Dataset'.format(mode.capitalize())]

    images = get_test_images(infer_dir, infer_img)
    dataset.set_images(images)
    imid2path = dataset.get_imid2path
    anno_file = dataset.get_anno()
    clsid2catid, catid2name = get_categories(cfg.metric, anno_file=anno_file)
    datainfo = {'imid2path': imid2path,
                'clsid2catid': clsid2catid,
                'catid2name': catid2name}
    _eval_batch_sampler = paddle.io.BatchSampler(
                dataset, batch_size=batch_size)
    loader = create('{}Reader'.format(mode.capitalize()))(
                dataset, 2, _eval_batch_sampler)
    # loader have images, step_id=0, data contain bbox, bbox_num, neck_feats
    return loader, datainfo

def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images

def ext_score(outs, data, datainfo, label_id):
    """
    extract the detection score of the learned adversarial sample.
    Args:
        outs(dict): output of the learned adversarial sample 
        data(dict): the learned adversarial sample
        datainfo(dict): data information of the specific dataset
        label_id(int): the original target class id 
    """

    clsid2catid = datainfo['clsid2catid']
    catid2name = datainfo['catid2name']
    for key in ['im_shape', 'scale_factor', 'im_id']:
        outs[key] = data[key]
    for key, value in outs.items():
        if hasattr(value, 'numpy'):
            outs[key] = value.numpy()

    batch_res = get_infer_results(outs, clsid2catid)
    start = 0
    flag = True
    bbox_num = outs['bbox_num']
    
    truck_num = 0
    for i, im_id in enumerate(outs['im_id']):
        end = start + bbox_num[i]
        bbox_res = batch_res['bbox'][start:end]
        for dt in numpy.array(bbox_res):
            catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
            
            if catid in [label_id, 3, 6] and score > 0.25:
                #print(im_id,  catid, catid2name[catid], score)
                truck_num += 1
                break
        
    return truck_num


def test():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    merge_config(FLAGS.opt)

    place = paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    test()
