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
Adversarial examples generation using patch attack algorithm.
Author: tianweijuan
"""

import paddle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
import numpy as np
import sys
sys.path.append("..")
import random
import cv2

from past.utils import old_div

def unpickle(file):

    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding = 'bytes')
    train_labels = dict[b'labels']
    train_array = dict[b'data']
    #train_array=train_array.tolist()

    fo.close()
    return train_labels, train_array

def main():
    #import pdb
    #pdb.set_trace()
    patchsize = 10
    adv_path = './detectors_adv/'
    # Load the model
    model = paddle.vision.models.resnet34(pretrained=False, num_classes=10)
    state_dict = paddle.load("./checkpoints/final.pdparams")
    model.set_state_dict(state_dict)
    # 获取测试数据
    temp_labels,temp_datas=unpickle("./cifar_data/" + "cifar-10-batches-py/test_batch")
    temp_labels=np.array(temp_labels)
    temp_datas=np.array(temp_datas)
    
    temp_datas = temp_datas.reshape((-1,3,32,32))#.transpose((0, 2, 3, 1))
    temp_datas = temp_datas / 255.
    
    temp_datas = temp_datas.astype("float32")
    Y_test = temp_labels
    X_test = temp_datas
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    means = np.tile(np.array(mean).reshape((1, 3, 1, 1)), [10000, 1, 32, 32])
    stds = np.tile(np.array(std).reshape((1, 3, 1, 1)), [10000, 1, 32, 32])
    
    X_test = old_div((X_test - means), stds) # 32, 32, 3


    adv_xs = []
    for idx in range(len(X_test)):
        x_test, y_test = X_test[idx:idx+1], Y_test[idx:idx+1]
        init_patch = paddle.fluid.initializer.Normal(loc=2.5, scale=0.8)

        patch = paddle.fluid.layers.create_parameter(x_test.shape, 'float32', default_initializer = init_patch)
        patch = paddle.clip(patch, min = x_test.min(), max = x_test.max())
        mask = paddle.zeros((x_test.shape[2], x_test.shape[3]))
       
        centerx = random.randint(10, 20)
        centery = random.randint(10, 20)
        mask[centerx-patchsize//2:centerx+patchsize//2, centery-patchsize//2:centery+patchsize//2] = 1
        patch = patch * mask
        x_test = paddle.to_tensor(x_test, dtype='float32', place= paddle.CUDAPlace(0))
        opt = paddle.optimizer.Adam(learning_rate=0.01, parameters = [patch])
        model.eval()
        for _ in range(100):
           
            input_x = x_test * (1-mask) + patch * mask
            predict = model(input_x)#[0]
            label = predict.numpy().argmax(axis=1)[0]           
            if label != y_test[0] or _ > 100:
                print('attack success, adversarial_label=%d' % label, y_test[0])
                break
            loss = predict[0, y_test[0]]
            loss.backward(retain_graph=True)
            opt.minimize(loss)
          
        patch = paddle.clip(patch, min = x_test.min(), max = x_test.max())
        adv_xs.append(x_test* (1-mask) + patch * mask)
    adv_xs = paddle.concat(adv_xs, axis=0)
    adv_xs = adv_xs.numpy() 
    adv_xs = adv_xs * stds + means
    adv_xs = adv_xs * 255.0
    adv_xs = np.clip(adv_xs, 0, 255).astype(np.uint8)
    adv_xs = adv_xs.transpose(0, 2, 3, 1)
    adv_file_path = "./detectors_adv/" + 'advcifar' + '_patch' + '.npy'
    np.save(adv_file_path, adv_xs) 
    return adv_xs

main()
