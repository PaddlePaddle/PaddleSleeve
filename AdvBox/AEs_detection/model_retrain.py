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
The image classification model training for cifar10 dataset with pretrained model.
Author: tianweijuan
"""

import paddle
from paddle.optimizer import Momentum
from paddle.vision import transforms as T
from paddle.vision.datasets import Cifar10
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform_train = T.Compose([T.Resize((32, 32)),
                             T.RandomHorizontalFlip(0.5),
                             T.RandomVerticalFlip(0.5),
                             T.Transpose(),
                             T.Normalize(
                                 mean=[0, 0, 0],
                                 std=[255, 255, 255]),
                             # output[channel] = (input[channel] - mean[channel]) / std[channel]
                             T.Normalize(mean=MEAN,
                                         std=STD)
                             ])
transform_eval = T.Compose([T.Resize((32, 32)),
                            T.Transpose(),
                            T.Normalize(
                                mean=[0, 0, 0],
                                std=[255, 255, 255]),
                            # output[channel] = (input[channel] - mean[channel]) / std[channel]
                            T.Normalize(mean=MEAN,
                                        std=STD)
                            ])

train_dataset = Cifar10(mode='train', transform=transform_train)
val_dataset = Cifar10(mode='test', transform=transform_eval)

network = paddle.vision.models.resnet34(pretrained=False, num_classes=10)
state_dict = paddle.load("./checkpoints/final.pdparams")

network.set_state_dict(state_dict)

model = paddle.Model(network)
optimizer = Momentum(learning_rate=0.001,
                     momentum=0.9,
                     weight_decay=L2Decay(1e-4),
                     parameters=model.parameters())
import pdb
pdb.set_trace()
# 进行训练前准备

earlystop = paddle.callbacks.EarlyStopping( 
    # acc不在上升时停止
    'acc',
    mode='max',
    patience=4,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)

model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
# 启动训练
model.fit(train_dataset,
          val_dataset,
          epochs=100, #时间原因只训练5轮
          batch_size=64,
          save_dir="./checkpoints/final/",
          num_workers=8,
          callbacks=[earlystop])




