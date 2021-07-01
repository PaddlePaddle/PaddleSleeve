#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
train a basic CNN model based on paddle 2
"""
import paddle
print(paddle.__version__)

import numpy as np
import matplotlib.pyplot as plt
import os
from paddle.vision.transforms import Compose, Normalize
from mnist_cnn_model import CNNModel

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')


train_data0, train_label_0 = train_dataset[0][0], train_dataset[0][1]
train_data0 = train_data0.reshape([28, 28])
# plt.figure(figsize=(2, 2))
# plt.imshow(train_data0, cmap=plt.cm.binary)
print('train_data0 label is: ' + str(train_label_0))

import paddle.nn.functional as F
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
# 加载训练集 batch_size 设为 64
def train(model):
    """

    Args:
        model: paddle model

    Returns:

    """
    model.train()
    epochs = 10
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            #tip：F.cross_entropy, 该OP实现了softmax交叉熵损失函数
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".\
                      format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()

    path = "./finetuing_cnn"    
    if not os.path.exists(path):
       os.makedirs(path)

    paddle.save(model.state_dict(), "finetuing_cnn/mnist_cnn.pdparams")
    paddle.save(optim.state_dict(), "finetuing_cnn/adam.pdopt")


#验证模型
test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)


# 加载测试数据集
def test(model):
    """

    Args:
        model: paddle model

    Returns:

    """
    model_path = 'finetuing_cnn/mnist_cnn.pdparams'
    if os.path.exists(model_path):
        para_state_dict = paddle.load(model_path)
        model.set_dict(para_state_dict)
        print('Loaded trained params of model successfully')
    else:
        print("model path not ok!")
        raise ValueError('The model_path is wrong: {}'.format(model_path))

    model.eval()
    batch_size = 64
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 20 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))



if __name__ == "__main__":
    model = CNNModel()
    train(model)
    test(model)

