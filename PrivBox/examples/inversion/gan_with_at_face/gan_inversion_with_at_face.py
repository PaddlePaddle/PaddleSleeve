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
This module provides an example of GAN attack on AT&T face dataset.
"""

from __future__ import print_function

import os

import argparse
import math
import numpy as np


from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import paddle
import paddle.nn.functional as F
from privbox.inversion import GANInversionAttack
from paddle import nn

from privbox.dataset import ATTFace
from paddle.vision.transforms import Compose, Normalize


# globel parameters

# resize face image to (img_size, img_size)
img_size = 64
# last class for fake label
target_fake_label = 40
# one more label dimension for target_fake_label
label_size = 41


def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("GAN")
    parser.add_argument("--batch_size",
                        type=int, default=32,
                        help="The batch size of training.")
    parser.add_argument("--attack_epoch",
                        type=int, default=100,
                        help="The iterations of attacking training.")
    parser.add_argument("--target_label",
                        type=int, default=1,
                        help="Attacked target face data label.")
    parser.add_argument("--learning_rate_real",
                        type=float, default=2e-4,
                        help="The learning rate of training Discriminator used real data.")
    parser.add_argument("--learning_rate_fake",
                        type=float, default=2e-4,
                        help="The learning rate of training Discriminator used fake data.")
    parser.add_argument("--learning_rate_gen",
                        type=float, default=2e-4,
                        help="The learning rate of training Generator used fake data.")
    parser.add_argument("--result_dir",
                        type=str, default="./att_results",
                        help="the directory for saving attack result.")
    parser.add_argument("--num_pic_save",
                        type=int, default=4,
                        help="Number of images save for each epoch. "
                        "Save first 'num_pic_save' pictures if batch size "
                        "greater than 'num_pic_save' for less memory usage.")

    args = parser.parse_args()
    return args


class GeneratorNet(nn.Layer):
    """
    Generative model, based on DCGAN (https://arxiv.org/pdf/1511.06434.pdf)
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        weight_attr = nn.initializer.Normal(0, 0.02)

        self.gen = nn.Sequential(
            nn.Conv2DTranspose(100, 512, (4, 4), (2, 2), (1, 1), weight_attr=weight_attr, bias_attr=False),
            nn.BatchNorm(512),
            nn.LeakyReLU(),
            nn.Conv2DTranspose(512, 256, (4, 4), (2, 2), (1, 1), weight_attr=weight_attr, bias_attr=False),
            nn.BatchNorm(256),
            nn.LeakyReLU(),
            nn.Conv2DTranspose(256, 128, (4, 4), (2, 2), (1, 1), weight_attr=weight_attr, bias_attr=False),
            nn.BatchNorm(128),
            nn.LeakyReLU(),
            nn.Conv2DTranspose(128, 64, (4, 4), (2, 2), (1, 1), weight_attr=weight_attr, bias_attr=False),
            nn.BatchNorm(64),
            nn.LeakyReLU(),
            nn.Conv2DTranspose(64, 1, (4, 4), (2, 2), (1, 1), weight_attr=weight_attr, bias_attr=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        forward computing
        """
        return self.gen(x)


class DiscriminatorNet(nn.Layer):
    """
    model for training AT&T face data
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.image_size = 64

        self.linear1 = nn.Linear(512, 400)
        self.linear2 = nn.Linear(400, 41)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

        self.conv_net = nn.Sequential(
            nn.Conv2D(1, 32, (5, 5)),
            nn.Tanh(),
            nn.MaxPool2D((3, 3), (3, 3)),
            nn.Conv2D(32, 64, (5, 5)),
            nn.Tanh(),
            nn.MaxPool2D((2, 2), (2, 2)),
            nn.Conv2D(64, 128, (5, 5)),
            nn.Tanh(),
            nn.MaxPool2D((2, 2), (2, 2))
        )

    def forward(self, x):
        """
        forward computing
        """
        x = paddle.reshape(x, [-1, 1, self.image_size, self.image_size])
        y = self.conv_net(x)
        y = paddle.reshape(y, [-1, 512])
        y = self.linear1(y)
        y = self.tanh(y)
        y = self.linear2(y)
        y = self.softmax(y)

        return y


class FakeDataset(paddle.io.Dataset):
    """
    Generate noise dataset for GeneratorNet
    """
    def __init__(self, data_size):
        self.data_size = data_size

    def __getitem__(self, idx):
        image = paddle.uniform([100, 2, 2]).astype('float32')
        label = paddle.uniform([1, label_size]).astype('float32')
        return image, label

    def __len__(self):
        return self.data_size


def plot(gen_data):
    """
    plot image data
    """
    pad_dim = 1
    paded = pad_dim + img_size
    gen_data = gen_data.reshape(gen_data.shape[0], img_size, img_size)
    n = int(math.ceil(math.sqrt(gen_data.shape[0])))
    gen_data = (np.pad(
        gen_data, [[0, n * n - gen_data.shape[0]], [pad_dim, 0], [pad_dim, 0]],
        'constant').reshape((n, n, paded, paded)).transpose((0, 2, 1, 3))
                .reshape((n * paded, n * paded)))
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(gen_data, cmap='Greys_r', vmin=-1, vmax=1)
    return fig


def show_image_grid(images, pass_id, result_dir, save_num):
    """
    save image
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_num_pic = images.shape[0]
    # save first save_num pic if batch size greater than save_num for less memory usage
    if save_num_pic > save_num:
        save_num_pic = save_num
    
    for i in range(save_num_pic):
        image = images[i].numpy().reshape([1, img_size, img_size])
        fig = plot(image)
        plt.savefig('{}/{:04d}_{:04d}.png'.format(result_dir, pass_id, i), bbox_inches='tight')
        plt.close(fig)


def train_and_attack(args):
    """
    run normal model training with GAN inversion attack

    Args:
        args(ArgumentParser): the execution parameters.
    """
    if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)

    # define AT&T dataset and dataloader, resize and normalize data
    resize_img = paddle.vision.transforms.Resize((img_size, img_size))
    normalize_img = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
    transform = Compose([resize_img, normalize_img])
    face_dataset = ATTFace(transform=transform)
    face_dataloader = paddle.io.DataLoader(face_dataset, batch_size=args.batch_size, shuffle=True)

    # define fake dataset
    fake_dataset = FakeDataset(10000)
    fake_dataset = paddle.io.DataLoader(fake_dataset, batch_size=args.batch_size)

    # define dicriminator (target model) and generator
    disc_net = DiscriminatorNet()
    gen_net = GeneratorNet()
    
    # define GAN attack and reconstruct target data
    gan_attack = GANInversionAttack(gen_net, label_size, args.target_label, target_fake_label, fake_dataset)

    gan_params = {"learning_rate_real": args.learning_rate_real,
                    "learning_rate_fake": args.learning_rate_fake,
                    "learning_rate_gen": args.learning_rate_gen}
    gan_attack.set_params(**gan_params)

    for pass_id in range(args.attack_epoch):
        real_acc_set = []
        real_avg_cost_set = []

        epoch = {"epoch": pass_id}

        acc_fn = paddle.metric.Accuracy()
        for data in face_dataloader():
            # Attacker: train discriminator and generator
            disc_net = gan_attack.fit(disc_net, None, **epoch)

            # Victim: train discriminator using real data
            real_image = data[0]
            real_batch_labels = data[1]
            real_batch_labels = real_batch_labels.reshape([real_batch_labels.shape[0]])
            real_labels = F.one_hot(real_batch_labels, label_size)

            p_real = disc_net(real_image)
            real_avg_cost = F.binary_cross_entropy_with_logits(p_real, real_labels)
            optim_d_real = paddle.optimizer.Adam(learning_rate=2e-4, parameters=disc_net.parameters())
            real_avg_cost.backward()
            optim_d_real.step()
            optim_d_real.clear_grad()
            
            real_acc = acc_fn.compute(p_real, real_labels)
            real_avg_cost_set.append(real_avg_cost.numpy()[0])
            real_acc_set.append(real_acc.numpy()[0])

        print("Victim Epoch: %d, real_avg_acc: %f, real_avg_cost: %f"
          % (pass_id, np.mean(real_acc_set), np.mean(real_avg_cost_set)))

        r_i = gan_attack.reconstruct()
        show_image_grid(r_i, pass_id, args.result_dir, args.num_pic_save)
        
    exit("Attack Finish")


if __name__ == "__main__":
    arguments = parse_args()
    print("args: ", arguments)
    train_and_attack(arguments)
