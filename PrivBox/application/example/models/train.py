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

import paddle
import argparse
from paddle.vision.transforms import Compose, Normalize


def parse_args():
    """
    Parse command line arguments.

    Returns:
        (ArgumentParser): command parse result
    """
    parser = argparse.ArgumentParser("ML_Leaks")
    parser.add_argument("--batch_size",
                        type=int, default=128,
                        help="The batch size of normal training.")
    parser.add_argument("--epoch",
                        type=int, default=10,
                        help="The iterations of target model training.")
    parser.add_argument("--lr",
                        type=float, default=0.0002,
                        help="The learning rate of target model training.")
    parser.add_argument("--model", type=str, choices=['resnet18', 'resnet34'],
                        default='resnet18',
                        help="training what model resnet18 or resnet34")
    args = parser.parse_args()
    return args


def train_resnet18_with_cifar10(args):
    """R
    train resnet18
    """
    transform = Compose([paddle.vision.Resize((32, 32)),
                         Normalize(mean=[127.5], std=[127.5], data_format='CHW'),
                         paddle.vision.transforms.Transpose()])
    net = paddle.vision.resnet18(num_classes=10)
    model = paddle.Model(net)
    data_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
    data_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform)


    model.prepare(
            paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters()),
            loss=paddle.nn.CrossEntropyLoss(),
            metrics=[paddle.metric.Accuracy()])

    callbacks = paddle.callbacks.EarlyStopping(
        'loss',
        mode='min',
        patience=2,
        verbose=1,
        min_delta=0.01,
        baseline=None,
        save_best_model=False)
    model.fit(train_data=data_train,
              eval_data=data_test,
              batch_size=args.batch_size,
              callbacks=[callbacks],
              epochs=args.epoch)
    model.save("./resnet18_10classes/resnet18")


if __name__ == "__main__":
    args = parse_args()
    if args.model == 'resnet18':
        train_resnet18_with_cifar10(args)
