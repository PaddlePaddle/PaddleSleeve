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
This module provides an example of use DGCMomentum optimizer to prevent DLG attack on MNIST.
"""

from __future__ import print_function

import sys
import os
sys.path.append('../../PrivBox/')

import argparse
import numpy
import numpy as np

import paddle
from PIL import Image
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F

from inversion import DLGInversionAttack
from metrics import MSE, Accuracy, PSNR, SSIM
from dgc_momentum_optimizer import DGCMomentum
from paddle.optimizer.momentum import Momentum

def to_bool_type(arg):
    """
    change bool argument to python bool type
    """
    in_s = str(arg).upper()
    if 'TRUE'.startswith(in_s):
       return True
    elif 'FALSE'.startswith(in_s):
       return False
    else:
       raise ValueError("argument must be 'False' or 'True' for bool type")


def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("DGCOptimizer")
    parser.add_argument("--train_epoch",
                        type=int, default=2,
                        help="The iterations of normal training.")
    parser.add_argument("--train_batch_size",
                        type=int, default=64,
                        help="The batch size of normal training.")
    parser.add_argument("--train_lr",
                        type=float, default=0.001,
                        help="The learning rate of attacking training.")
    parser.add_argument("--sparsity",
                        type=float, default=0.8,
                        help="The learning rate of normal training.")
    parser.add_argument("--use_dgc",
                        type=to_bool_type, default=True,
                        help=" whether use DGCMomentum optimizer")
    parser.add_argument("--dlg_attack",
                        type=to_bool_type, default=False,
                        help=" whether launch dlg attack")

    """
    following arguments have same meaning with dlg attack example.
    see directory 'PrivBox/example/inversion/dlg_with_mnist' for details.
    """
    parser.add_argument("--attack_batch_size",
                        type=int, default=1,
                        help="The batch size of dlg attack training.")
    parser.add_argument("--attack_epoch",
                        type=int, default=2000,
                        help="The iterations of attacking training.")
    parser.add_argument("--attack_lr",
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
    train linear model with DGCMomentum optimizer or Momentum optimizer.
    Additionally, you can launch a dlg attack when training.

    Args:
        args(ArgumentParser): the execution parameters.
    """
    if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)

    launch_attack = args.dlg_attack
    # a small batch size for dlg attack
    if launch_attack:
        args.train_batch_size = args.attack_batch_size
    
    # load mnist data
    transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # define a linear network
    net = LinearNet()
    
    # define optimizer
    opt = Momentum(learning_rate=args.train_lr, parameters=net.parameters())
    if args.use_dgc:
        opt = DGCMomentum(learning_rate=args.train_lr,
                parameters=net.parameters(), sparsity=args.sparsity, use_nesterov=False)
    
    # train model
    for i in range(args.train_epoch):
        acc = paddle.metric.Accuracy()
        for id, d in enumerate(train_loader()):
            x = d[0].reshape((len(d[0]), 1, 28 * 28))
            y = F.one_hot(d[1], 10)
            predict = net(x)
            # only the mse loss is now supported for dlg attack
            if launch_attack: 
                loss = F.mse_loss(predict, y, reduction="none")
            else:
                loss = F.cross_entropy(predict, d[1])
            loss.backward()
            acc_ = acc.compute(predict.squeeze(), d[1])
            acc.update(acc_)
            params_grads = opt.backward(loss, parameters=net.parameters())

            # launch attack
            if launch_attack:
                grads = [g[1] for g in params_grads]
                for data_idx in range(len(x)):
                    image = Image.fromarray(((x[data_idx].numpy() + 1) * 127.5).reshape(28, 28).astype(numpy.uint8))
                    image.save(args.result_dir + "/target_{}.png".format(data_idx))
                attack(args, grads, x, y, net)
                launch_attack = False
                return
            # update network
            opt.apply_gradients(params_grads)
            opt.clear_grad()
            if id % 100 == 0:
                print("epoch {}, batch id {}, training loss {}, acc {}."
                       .format(i, id, float(paddle.mean(loss)), acc.accumulate()))
    

def attack(args, origin_grad, target_x, target_y, net):
    """
    dlg attack
    """
    # define DLG attack and reconstruct target data
    dlg_attack = DLGInversionAttack(net, origin_grad, target_x.shape, target_y.shape)

    dlg_params = {"learning_rate": args.attack_lr,
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


if __name__ == "__main__":
    arguments = parse_args()
    print("args: ", arguments)
    train_and_attack(arguments)
