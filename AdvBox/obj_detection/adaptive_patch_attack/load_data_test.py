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
The patch position and size determination for all testing images.
Contains:
* PatchTransformer: transforms batch of patches.
* PatchApplier: applies adversarial patches to testing images.

Author: tianweijuan
"""

import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image

class MedianPool2d(nn.Layer):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
         or non-local smoothing filter using non-local mean denoiser
    """
   
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        
        self.k = (kernel_size, kernel_size) #_pair(kernel_size)
        self.stride = (stride, stride)#_pair(stride)
        self.padding = (padding, padding, padding, padding)#_quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        
        if self.same:
            ih, iw = x.shape[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        
        x = F.pad(x, self._padding(x), mode='reflect')
        #x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        _, c, h, w = x.shape
        x = F.unfold(x, [self.k[0], self.k[1]], self.stride[0], 0, 1)
        h_out =  h - self.k[0] + 1
        w_out = w - self.k[1] + 1
        x = paddle.transpose(paddle.reshape(x, [1, c, self.k[0] * self.k[1], h_out, w_out]), [0, 1, 3, 4, 2])
        
        x = paddle.median(x, axis=-1)
        return x




class PatchTransformer(nn.Layer):
    """PatchTransformer: transforms batch of patches
    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7,same=True)


    def forward(self, adv_patch, lab_batch, img_size, patch_scale, patchxy_scale, do_rotate=True, rand_loc=True):
        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        
        adv_patch_pool = self.medianpooler(adv_patch)
        # Determine size of padding
        pad_h = (img_size[0] - adv_patch_pool.shape[-2]) / 2
        pad_w = (img_size[1] - adv_patch_pool.shape[-1]) / 2
        # Make a batch of patches
        
        adv_patch = adv_patch.unsqueeze(0)#.unsqueeze(0)
        adv_batch = paddle.expand(adv_patch, [lab_batch.shape[0], lab_batch.shape[1], -1, -1, -1])
        batch_size = [lab_batch.shape[0], lab_batch.shape[1]]
        
        # Contrast, brightness and noise transforms
        
        # Create random contrast tensor
        contrast = paddle.uniform(shape=batch_size, min=self.min_contrast, max=self.max_contrast)
        #paddle.to_tensor(batch_size, dtype=float32, palce=CUDAPlace(0)).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = paddle.expand(contrast, [-1, -1, adv_batch.shape[-3], adv_batch.shape[-2], adv_batch.shape[-1]])
        #contrast = contrast.cuda()
        

        # Create random brightness tensor
        brightness =  paddle.uniform(shape=batch_size, min=self.min_brightness, max=self.max_brightness) 
        
        #paddle.to_tensor(batch_size,  dtype=float32, palce=CUDAPlace(0)).uniform_(self.min_brightness, self.max_brightness)
        
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = paddle.expand(brightness, [-1, -1, adv_batch.shape[-3], adv_batch.shape[-2], adv_batch.shape[-1]])
        #brightness = brightness.cuda()
    
        noise = paddle.uniform(shape=adv_batch.shape, min=-1, max=1) * self.noise_factor
        #paddle.to_tensor(adv_batch.shape, dtype=float32, palce=CUDAPlace(0)).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        #adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = paddle.clip(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = paddle.slice(lab_batch, [2], [0], [1])
        cls_mask = paddle.expand(cls_ids, [-1, -1, 3])
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = paddle.expand(cls_mask, [-1, -1, -1, adv_batch.shape[3]])
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = paddle.expand(cls_mask, [-1, -1, -1, -1, adv_batch.shape[4]])
        
        msk_batch = paddle.full(shape = cls_mask.shape, fill_value=7) - cls_mask
        
        
        #paddle.to_tensor(cls_mask.size(), dtype=float32, palce=CUDAPlace(0)).fill_(3) - cls_mask
        
        # Pad patch and mask to image dimensions
        
        mypad = paddle.nn.Pad2D((int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h)), mode='constant', value=0)
        #nn.ConstantPad2d((int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h)), 0) # 左右上下
        s = adv_batch.shape
        adv_batch = paddle.reshape(adv_batch, [s[0]*s[1], s[2], s[3], s[4]])
        msk_batch = paddle.reshape(msk_batch, [s[0]*s[1], s[2], s[3], s[4]])
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = (lab_batch.shape[0] * lab_batch.shape[1])
        
        if do_rotate:
            angle = paddle.uniform(shape=[anglesize], min=self.minangle, max=self.maxangle)
        else: 
            angle = paddle.full([anglesize], fill_value=0)

        # Resizes and rotates
        current_patch_size_w, current_patch_size_h = adv_patch.shape[-1], adv_patch.shape[-2]
        lab_batch_scaled = paddle.full(lab_batch.shape, fill_value=0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size[1]
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size[0]
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size[1]
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size[0]
        #target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        # patch 大小占目标框大小的比例
        # patch 坐标占目标框坐标之间的比例
        
        target_size_x =  lab_batch_scaled[:, :, 3] * (patchxy_scale[0]) + (lab_batch_scaled[:, :, 1] - lab_batch_scaled[:, :, 3] /2.) 
        target_size_y =  lab_batch_scaled[:, :, 4] * (patchxy_scale[1]) + (lab_batch_scaled[:, :, 2] - lab_batch_scaled[:, :, 4] /2.)
            
        target_size_w = (lab_batch_scaled[:, :, 3] * (patch_scale[0])) #0.39
        target_size_h = (lab_batch_scaled[:, :, 4] * (patch_scale[1])) #0.46
        
        target_x = lab_batch[:, :, 1].reshape([np.prod(batch_size)])
        target_y = lab_batch[:, :, 2].reshape([np.prod(batch_size)])
        targetoff_x = lab_batch[:, :, 3].reshape([np.prod(batch_size)])
        targetoff_y = lab_batch[:, :, 4].reshape([np.prod(batch_size)])
        if(rand_loc):
            off_x = targetoff_x* (paddle.uniform(targetoff_x.shape, min=-0.4, max=0.4))
            target_x = target_x + off_x
            off_y = targetoff_y*(paddle.uniform(targetoff_y.shape, min=-0.4, max=0.4))
            target_y = target_y + off_y
        #target_y = target_y - 0.05
         
         
        scale_w, scale_h = target_size_w / current_patch_size_w, target_size_h / current_patch_size_h# determine the patch size =  w^2 + h^2 / 100 
        scale_w, scale_h =  scale_w.reshape([anglesize]), scale_h.reshape([anglesize])
        
        scale_target_w, scale_target_h = (target_size_w /img_size[1]).reshape([anglesize]), (target_size_h / img_size[0]).reshape([anglesize])
        
        scale_patch_xmin, scale_patch_ymin = (target_size_x /img_size[1]).reshape([anglesize]), (target_size_y / img_size[0]).reshape([anglesize]) 
    
        
        #tx = (-target_x+0.5)*2
        #ty = (-target_y+0.5)*2
        
        tx = (-scale_patch_xmin - scale_target_w/2. + 0.5) * 2
        ty = (-scale_patch_ymin - scale_target_h/2. + 0.5) * 2
               
        sin = paddle.sin(angle)
        cos = paddle.cos(angle)        
        
        # Theta = rotation,rescale matrix
        theta = paddle.full([anglesize, 2, 3], fill_value=0)
        theta[:, 0, 0] = cos/scale_w
        theta[:, 0, 1] = sin/scale_h
        theta[:, 0, 2] = tx*cos/scale_w+ty*sin/scale_h
        theta[:, 1, 0] = -sin/scale_w
        theta[:, 1, 1] = cos/scale_h
        theta[:, 1, 2] = -tx*sin/scale_w+ty*cos/scale_h

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)
        

        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2
        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)
        '''
        
        s1 = adv_batch_t.shape
        adv_batch_t = adv_batch_t.reshape([s[0], s[1], s[2], s1[-2], s1[-1]])
        msk_batch_t = msk_batch_t.reshape([s[0], s[1], s[2], s1[-2], s1[-1]])

        adv_batch_t = paddle.clip(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        #img.show()
        #exit()

        return adv_batch_t * msk_batch_t, adv_patch_pool

class PatchApplier(nn.Layer):
    """PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        
        advs = paddle.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = paddle.where((adv == 0), img_batch, adv)
        return img_batch





