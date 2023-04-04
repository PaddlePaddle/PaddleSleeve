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
The adaptive patch attack apply for testing dataset with acquired adversarial patch.
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
from load_data_test import *
import xmltodict
import math
import random
from past.utils import old_div
from paddle.vision import transforms

batch_size = 1
epochs = 1

logger = setup_logger('train')
def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    
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
        default=0.4,
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

    def __init__(self, FLAGS, dic, h, w):
        super(AttackNet, self).__init__()

        mask_list = dic['annotation']['object']
        box_list = dic['annotation']['size']
        obj_list = dic['annotation']['obj_size']
        self.widtht, self.heightt = int(box_list['width']), int(box_list['height'])
        self.obj_xmin, self.obj_ymin, self.obj_w, self.obj_h = float(obj_list['xmin']), float(obj_list['ymin']), float(obj_list['width']), float(obj_list['height'])
        
        self.xmin, self.ymin, self.xmax, self.ymax = get_mask_coordination(mask_list)
        self.h, self.w = h, w 
        self.patch_scale_w, self.patch_scale_h = (self.xmax -self.xmin) / self.obj_w, (self.ymax- self.ymin) / self.obj_h
        self.patch_scale_xmin, self.patch_scale_ymin  = (self.xmin - self.obj_xmin)/self.obj_w, (self.ymin - self.obj_ymin) / self.obj_h

        transform = transforms.ToTensor()
        patch_img0 = transform(Image.open(FLAGS.output_patch_dir+"/adv_patch0.png").convert('RGB')).cuda()
               
        self.patch_img0 = patch_img0.unsqueeze(0) 
        
       
        self.patch_transformer = PatchTransformer()
        self.patch_applier = PatchApplier()





    def forward(self, input1, epoch):
        
        orig_img = input1['image_org'] # 查看输出是tensor
        #input1.pop['image_org']
        orig_img = paddle.cast(orig_img, dtype='float32')
        scale = 1.0 / 255.0
        orig_img *= scale

        adv_batch_t0, _ = self.patch_transformer(self.patch_img0, input1["label"], (self.heightt, self.widtht), (self.patch_scale_w, self.patch_scale_h), (self.patch_scale_xmin, self.patch_scale_ymin), do_rotate=False, rand_loc=False)
        #import pdb
        #pdb.set_trace()
        orig_img = orig_img.transpose((0, 3, 1, 2))
        p_img_batch = self.patch_applier(orig_img, adv_batch_t0)
        if not os.path.exists("./patch_test"):
            os.makedirs("./patch_test")
        for i in range(batch_size):
            img = cv2.cvtColor(p_img_batch[i].transpose((1, 2, 0)).detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite("./patch_test/"+ input1["im_file"][i].split("/")[-1], img)
        
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
    model_attack = AttackNet(FLAGS, dic, 320, 320)
    for epoch in range (epochs):
        try:
            for step_id, data in enumerate(loader):
            
                model_attack(data, epoch)
        except StopIteration:
            print ('here is end')




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

    #loader = create('TestReader')(dataset, 0)
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
    # loader have  images, step_id=0, data contain bbox, bbox_num, neck_feats

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
