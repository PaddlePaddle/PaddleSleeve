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
This module provides an example of DLG attack on MNIST.
"""

from __future__ import print_function

import os

import argparse
import numpy
import numpy as np

import paddle
from PIL import Image
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F

from privbox.inversion import DLGInversionAttack
from privbox.metrics import MSE, Accuracy, PSNR, SSIM


def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("DLG")
    parser.add_argument("--batch_size",
                        type=int, default=1,
                        help="The batch size of normal training.")
    parser.add_argument("--attack_epoch",
                        type=int, default=2000,
                        help="The iterations of attacking training.")
    parser.add_argument("--learning_rate",
                        type=float, default=0.2,
                        help="The learning rate of attacking training.")
    parser.add_argument("--result_dir",
                        type=str, default="./att_results",
                        help="the directory for saving attack result.")
    parser.add_argument("--return_epoch",
                        type=int, default=100,
                        help="return reconstructed data every 'return_epoch' epochs.")
    parser.add_argument("--window_size",
                        type=int, default=200,
                        help="when batch size greater than 1, "
                              "we update single data roundly for each window size iterations")
    args = parser.parse_args()
    return args


class LinearNet(paddle.nn.Layer):
    """
    Define a Linear Network for MNIST
    """
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear = paddle.nn.Linear(28 * 28, 10)

    def forward(self, x):
        """
        Override forward computing
        """
        y = self.linear(x)
        return y


def train_and_attack(args):
    """
    The training procedure that starts from several normal training steps as usual,
    but entrance the dlg method as soon as the gradients of target data are obtained.

    Args:
        args(ArgumentParser): the execution parameters.
    """
    if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)
    
    # load mnist data and define target data
    transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    data = train_loader().next()
    target_x = data[0]
    target_y = data[1]

    for data_idx in range(len(target_x)):
        image = Image.fromarray(((target_x[data_idx].numpy() + 1) * 127.5).reshape(28, 28).astype(numpy.uint8))
        image.save(args.result_dir + "/target_{}.png".format(data_idx))

    target_x = target_x.reshape((len(target_x), 1, 28 * 28))
    target_y = F.one_hot(target_y, 10)
    
    # define target net and get training gradients
    net = LinearNet()

    prediction = net(target_x)
    loss = F.mse_loss(prediction, target_y, reduction='none')

    origin_grad = paddle.grad(loss, net.parameters())
    
    # define DLG attack and reconstruct target data
    dlg_attack = DLGInversionAttack(net, origin_grad, target_x.shape, target_y.shape)

    dlg_params = {"learning_rate": args.learning_rate,
                    "attack_epoch": args.attack_epoch,
                    "window_size": args.window_size,
                    "return_epoch": args.return_epoch}

    dlg_attack.set_params(**dlg_params)

    ret = dlg_attack.reconstruct()

    # evaluate DLG attack and save result
    for i in range(len(ret)):
        rec_data = ret[i][0]
        rec_labels = ret[i][1]

        eval_x = dlg_attack.evaluate(target_x.reshape([-1, 1, 28, 28]),
                                     rec_data.reshape([-1, 1, 28, 28]),
                                     [MSE(), PSNR(), SSIM()])
        y_loss = dlg_attack.evaluate(target_y, rec_labels, [Accuracy()])[0]

        iteration = i * args.return_epoch

        print("Attack Iteration {}: data_mse_loss = {}, data_psnr = {}, data_ssim = {}, labels_acc = {}"
            .format(iteration, eval_x[0], eval_x[1], eval_x[2],
            y_loss))
        
        save_shape = (28, 28)
        for data_idx in range(rec_data.shape[0]):
            img = Image.fromarray(((rec_data[data_idx].numpy() + 1) * 127.5)
                                .reshape(save_shape)
                                .astype(numpy.uint8))
            img.save(args.result_dir + "/result_{}_{}.png".format(iteration, data_idx))
    exit("Attack Finish")


if __name__ == "__main__":
    arguments = parse_args()
    print("args: ", arguments)
    train_and_attack(arguments)
