# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
The ETO Attack algorithm for OCR recognition are implemented based on an improved pgd attack algorithm.
Contains:
* Initialize a OCR recognition model and inference pictures.
* Generate adversarial pertuation image using model weights with combined eot and pgd attack algorithm.
* Generate adversarial image.
Author: tianweijuan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json
import cv2
import math
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
from EOT_simulation import transformation


def ext_resize(image_shape, img):
    """
    extract the resized image size used for number and words recognition model.
    Args:
        image_shape: the origin image size
        img: the input image
    Return:
        the computed resized image size
    """
    _, imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    return resized_w, imgH


def ext_resize_chinese(image_shape, img):
    """
    extract the resized image size used for number, words and chinese recognition model.
    Args:
        image_shape: the origin image size
        img: the input image
    Return:
        the computed resized image size
    """
    _, imgC, imgH, imgW = image_shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    return resized_w, imgH


def main():
    global_config = config['Global']
    useEOT = True
    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                config['Architecture']["Models"][key]["Head"][
                    'out_channels'] = char_num
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

    model = build_model(config['Architecture'])

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global'].get('save_res_path',
                                         "./output/rec/predicts_rec.txt")
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()

    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
                img_cv = cv2.imread(file)
                
            batch = transform(data, ops)
            
            # CRNN
            if config['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    paddle.to_tensor(encoder_word_pos_list),
                    paddle.to_tensor(gsrm_word_pos_list),
                    paddle.to_tensor(gsrm_slf_attn_bias1_list),
                    paddle.to_tensor(gsrm_slf_attn_bias2_list)
                ]
            if config['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio)]

            images = np.expand_dims(batch[0], axis=0)
            EOT_transforms = transformation.target_sample()
            num_of_EOT_transforms = len(EOT_transforms)
            transform_eot = np.array(EOT_transforms).reshape(((94, 2, 3)))
            transform_eot = paddle.to_tensor(transform_eot, dtype= 'float32')
            images = paddle.to_tensor(images)
            
            images_batch = images
            for i in range(num_of_EOT_transforms - 1):
                images_batch = paddle.concat([images_batch, images], 0)    
            
            # add pgd attack algorithm                          
            epsilon_ball = 0.2
            steps = 200
            epsilon_stepsize = 1e-2 
            shape = images.shape
            resize_image_w, resize_image_h = ext_resize_chinese(shape, img_cv)
            grid = paddle.nn.functional.affine_grid(transform_eot, images_batch[:, :, :resize_image_h, :resize_image_w].shape)
            adv_img = paddle.to_tensor(images[:, :, :resize_image_h, :resize_image_w], dtype='float32', place = paddle.CUDAPlace(0))
            
            if useEOT == True:
                adv_img_batch = adv_img
                for i in range(num_of_EOT_transforms):
                    if i == num_of_EOT_transforms - 1: break
                    adv_img_batch = paddle.concat([adv_img_batch, adv_img], 0)
            else:
                adv_img_batch = adv_img
            
            adv_img_batch.stop_gradient = False
            
            _, imgC, imgH, imgW = shape
            
            padding_im = paddle.zeros([num_of_EOT_transforms, imgC, imgH, imgW - resize_image_w])
            padding_im.stop_gradient = True   
            adv_img_batch = paddle.nn.functional.grid_sample(adv_img_batch, grid, mode='bilinear')
            adv_img_concat = paddle.tanh(paddle.concat([adv_img_batch, padding_im], axis = 3))
            for step in range(steps):
                                
                adv_img_concat.stop_gradient = False
                if config['Architecture']['algorithm'] == "SRN":
                    preds = model(adv_img_concat, others)
                elif config['Architecture']['algorithm'] == "SAR":
                    preds = model(adv_img_concat, img_metas)
                else:
                    preds = model(adv_img_concat) # 1, 80, 6625
                preds_idx = preds.argmax(axis=2)
                
                probs = preds.max(axis=2)        
                
                prob_cond = paddle.where(preds_idx>0, probs, preds_idx.astype("float32"))
                loss = paddle.fluid.layers.reduce_sum(prob_cond) / num_of_EOT_transforms
                print('Step:', step, '======loss:', loss.numpy())
                loss.backward(retain_graph = True) 
                gradient = adv_img_concat.grad 
               
                if gradient.isnan().any():
                    paddle.assign(0.001 * paddle.randn(gradient.shape), gradient)
                
                normalized_gradient = paddle.sign(gradient)[:, :, :resize_image_h, :resize_image_w]
                eta = epsilon_stepsize * normalized_gradient
                
                adv_img_batch = adv_img_batch.detach() + eta.detach()
                eta = paddle.clip(adv_img_batch - images_batch[:, :, :resize_image_h, :resize_image_w], -epsilon_ball, epsilon_ball)
                adv_img_patch = images_batch[:, :, :resize_image_h, :resize_image_w] + eta
                adv_img_patch = paddle.clip(adv_img_patch, -1., 1.)
                adv_img_concat = paddle.tanh(paddle.concat([adv_img_patch, padding_im], axis = 3))
                
            preds = model(adv_img_concat)
            adv_img_patch = (adv_img_patch * 0.5 + 0.5) * 255.
            adv_img_vis = (adv_img_patch[0].transpose((1, 2, 0))).numpy()
            adv_img_vis = cv2.resize(adv_img_vis, (img_cv.shape[1], img_cv.shape[0]), cv2.INTER_CUBIC)
            cv2.imwrite("./output/adv_img_vis.png", adv_img_vis)
        
            post_result = post_process_class(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                        "label": post_result[key][0][0],
                        "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info)
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0] + "\t" + str(post_result[0][1])

            if info is not None:
                logger.info("\t result: {}".format(info))
                fout.write(os.path.basename(file) + "\t" + info + "\n")
        logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
