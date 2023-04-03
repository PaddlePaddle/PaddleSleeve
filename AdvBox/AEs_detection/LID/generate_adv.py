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
Adversarial examples generation using pgd attack algorithm.
Author: tianweijuan
"""

import paddle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
import numpy as np
import sys
sys.path.append("../..")
from adversary import Adversary
from attacks.gradient_method import PGD

from models.whitebox import PaddleWhiteBoxModel
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
    adv_path = './detectors_adv/'
    # Load the model
    model = paddle.vision.models.resnet34(pretrained=False, num_classes=10)
    state_dict = paddle.load("./checkpoints/final.pdparams")
    model.set_state_dict(state_dict)
  
    temp_labels,temp_datas=unpickle("./cifar_data/" + "cifar-10-batches-py/test_batch")
    temp_labels=np.array(temp_labels)
    temp_datas=np.array(temp_datas)
    
    temp_datas = temp_datas.reshape((-1,3,32,32))#.transpose((0, 2, 3, 1))
    temp_datas = temp_datas / 255.
    
    temp_datas = temp_datas.astype("float32")
    Y_test = temp_labels
    X_test = temp_datas
    X_test = paddle.to_tensor(X_test, dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    means = paddle.tile(paddle.to_tensor(mean).reshape((1, 3, 1, 1)), [10000, 1, 32, 32])
    stds = paddle.tile(paddle.to_tensor(std).reshape((1, 3, 1, 1)), [10000, 1, 32, 32])
    
    X_test = old_div((X_test - means), stds) # 32, 32, 3
    
    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [model],
        [1],
        (0, 1),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        input_channel_axis=0,
        input_shape=(3, 32, 32),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=10)
    attack = PGD(paddle_model, norm='Linf', epsilon_ball=16 / 255, epsilon_stepsize=2 / 255) # 16/255
    adv_data = []
    for i in range(len(X_test)):
        # non-targeted attack
        attack = PGD(paddle_model, epsilon_ball=16 / 255)
        x_test, y_test = X_test[i], Y_test[i]
        
        #x_test = x_test.transpose(2, 0, 1)
        x_test = paddle.unsqueeze(x_test, 0)
        
        predict = model(x_test)#[0]
        label = predict.numpy().argmax(axis=1)[0]
        x_test = np.squeeze(x_test)
        inputs = x_test
        labels = y_test
        adversary = Adversary(inputs.numpy(), labels)
    
        # 设定epsilons
        attack_config = {"steps": 20}
        adversary = attack(adversary, **attack_config)
        
        if adversary.is_successful():
              
            print('attack success, adversarial_label=%d'
              % adversary.adversarial_label)
            adv = adversary.adversarial_example
        else:
            adv = x_test.numpy()
        adv = np.squeeze(adv)
        
        adv = adv * np.reshape(np.array(std), [3, 1, 1]) + np.reshape(np.array(mean), [3, 1, 1])
        adv = adv.transpose(1, 2, 0)
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        adv_cv = np.copy(adv)
        adv_cv = adv_cv[..., ::-1]
        adv_data.append(adv_cv)
        
    adv_data = np.array(adv_data) 
    adv_file_path = adv_path + 'adv_datacifar' + '_pgdinf' + '.npy'
    np.save(adv_file_path, adv_data)

main()
