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
The image classification model finetune using mixture gaussian loss with pretrained baseline classification model.
Author: tianweijuan
"""

import paddle
from paddle.optimizer import Momentum
from paddle.vision import transforms as T
from paddle.vision.datasets import Cifar10
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
import numpy as np

class LgmLoss(paddle.nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        means_init = paddle.fluid.initializer.XavierInitializer(uniform = True)
        self.means = paddle.fluid.layers.create_parameter([self.num_classes, 10], 'float32', name="rbf_centers", default_initializer=means_init)
    def forward(self, y_pred, y_true):
        #print('built lgm inference')
        #feat = get_model(image, resnet_size, is_training, num_classes=num_classes, reuse=reuse, output_feat=True) 
        logits, likelihood_reg_loss = self.lgm_logits(y_pred, self.num_classes, labels=y_true, alpha=0.1, lambda_=0.01, batch_size = 200)
        y_true =  paddle.reshape(y_true, [200, 1])
        #y_true = paddle.nn.functional.one_hot(y_true, self.num_classes, name=None)
        cross_entropy_loss = paddle.fluid.layers.reduce_sum(paddle.nn.functional.softmax_with_cross_entropy(y_pred, y_true), dim=0)
              
        return (likelihood_reg_loss + cross_entropy_loss)/200.
    def lgm_logits(self, feat, num_classes, labels=None, alpha=0.1, lambda_=0.01, batch_size = 200):
        '''
        The 3 input hyper-params are explained in the paper.\n
        Support 2 modes: Train, Validation\n
        (1)Train:\n
        return logits, likelihood_reg_loss\n
        (2)Validation:\n
        Set labels=None\n
        return logits\n
        '''
        
        N = batch_size#feat.get_shape().as_list()[0]
        feat_len = feat.shape[1]
            
        #means = tf.get_variable('rbf_centers', [num_classes, feat_len], dtype=tf.float32,
        #                            initializer=tf.contrib.layers.xavier_initializer())
        XY = paddle.fluid.layers.matmul(feat, self.means, transpose_y=True)
        XX = paddle.fluid.layers.reduce_sum(paddle.square(feat), dim=1, keep_dim=True)
        
        YY = paddle.fluid.layers.reduce_sum(paddle.square(paddle.transpose(self.means, perm=[1, 0])), dim=0, keep_dim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        if labels is None:
            # Validation mode
            psudo_labels = paddle.fluid.layers.argmax(neg_sqr_dist, axis=1)
            means_batch = paddle.fluid.layers.gather(self.means, psudo_labels)
            likelihood_reg_loss = lambda_ * paddle.fluid.layers.reduce_sum((feat - means_batch) ** 2)/2. # paddle.fluid.layers.reduce_mean
            
            # In fact, in validation mode, we only need to output neg_sqr_dist.
            # The likelihood_reg_loss and means are only for research purposes.
            return neg_sqr_dist, likelihood_reg_loss, self.means
        # *(1 + alpha)
        labels_hot = paddle.nn.functional.one_hot(labels, self.num_classes, name=None)
        ALPHA = labels_hot * alpha
        #ALPHA = tf.one_hot(labels, num_classes, on_value=alpha, dtype=tf.float32)
        K = ALPHA + paddle.fluid.layers.ones([N, num_classes], dtype='float32')
        logits_with_margin = paddle.fluid.layers.elementwise_mul(neg_sqr_dist, K)
        # likelihood regularization

        labels_org = []
        for i in range(N):
            labels_org.append(np.argmax(labels[i]))
        labels_org =  paddle.to_tensor(labels_org, 'int32')
    
        means_batch = paddle.fluid.layers.gather(self.means, paddle.cast(labels_org, dtype='int32'))
        likelihood_reg_loss = lambda_ *  paddle.fluid.layers.reduce_sum((feat - means_batch) ** 2)/2.
        #print('LGM loss built with alpha=%f, lambda=%f\n' %(alpha, lambda_))
        return logits_with_margin, likelihood_reg_loss

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
optimizer = Momentum(learning_rate=0.01,
                     momentum=0.9,
                     weight_decay=L2Decay(1e-4),
                     parameters=model.parameters())

earlystop = paddle.callbacks.EarlyStopping( 
    'acc',
    mode='max',
    patience=4,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)

model.prepare(optimizer, LgmLoss(10), Accuracy(topk=(1, 5)))

model.fit(train_dataset,
          val_dataset,
          epochs=20, 
          batch_size=200,
          save_dir="./checkpoints/final_lgm/",
          num_workers=8,
          callbacks=[earlystop])







