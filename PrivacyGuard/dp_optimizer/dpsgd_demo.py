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
DPSGD demo.
"""

import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize
import time
import math
import dpoptimizer
import privacy_analysis
from paddle.io import Dataset, RandomSampler, BatchSampler


class Net(paddle.nn.Layer):
    """
    model
    """
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = paddle.nn.Linear(in_features=28 * 28, out_features=84)
        self.linear2 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        forward
        """
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

epochs = 10
batch_size = 64
N = 60000
l2_norm_clip = 10.0
sigma = 0.1
delta = 1e-5
guassian_scaler = l2_norm_clip * sigma

T = epochs * math.floor(N / batch_size)

# compute privacy budget
epsilon = privacy_analysis.compute_privacy(N=N, batch_size=batch_size, 
    T=T, sigma=sigma, delta=delta)
print("privacy analysis, epsilon={}".format(epsilon))

# batched random sampler
sampler = RandomSampler(data_source=train_dataset, replacement=True, num_samples=N)
batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
train_loader = paddle.io.DataLoader(train_dataset, batch_sampler=batch_sampler)


def train(model):
    """
    train model
    """
    parameters = model.parameters()
    clip = paddle.nn.ClipGradByNorm(clip_norm=l2_norm_clip)
    optim = dpoptimizer.DPSGD(learning_rate=0.001, parameters=parameters, 
        grad_clip=clip, stddev=guassian_scaler)

    steps = 0
    for epoch_id in range(epochs):
        start_time = time.time()
        for batch_id, data in enumerate(train_loader()):
            steps += 1
            x_data = data[0] # image
            y_data = data[1] # label
            predicts = model(x_data)
            loss = F.softmax_with_cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)

            if batch_id % 300 == 0:
                end_time = time.time()

                epsilon = privacy_analysis.compute_privacy(N=N, 
                    batch_size=batch_size, T=steps, sigma=sigma, delta=delta)
                print("epoch_id={}, batch_id={}, acc={}, (ε={:.2f}, δ={}), cost time={:.2f}s".format(
                    epoch_id, batch_id, acc.numpy(), epsilon, delta, (end_time - start_time)))
                start_time = time.time()
            
            optim.minimize(loss=loss, batch_size=batch_size)
            optim.clear_grad()


model = Net()
train(model)
