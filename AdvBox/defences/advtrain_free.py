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
paddle2 model for adversarial training based on Free-Advtrain.
* https://arxiv.org/abs/1904.12843
"""
import os
import sys

sys.path.append("../..")

import paddle
import numpy as np
import gc
import paddle.vision.transforms as T
import paddle.distributed as dist
import time
from defences.utils import *

logger = setup_logger('Free-AdvTrain')


def free_advtrain(model, train_dataset, test_dataset=None, save_path=None, **kwargs):
    """
    A demo for adversarial training based on Free-AT.
    Reference Implementation: https://arxiv.org/abs/1904.12843
    Adversarial Weight Perturbation Helps Robust Generalization.

    Author: Ali Shafahi.
    Args:
        model (paddle.nn.Layer): paddle model
        train_loader: paddle dataloader.
        test_loader: paddle dataloader.
        save_path: str. path for saving model.
        **kwargs: Other named arguments.
    Returns:
        training log
    """
    assert save_path is not None
    epoch_num = kwargs["epoch_num"]
    batch_size = kwargs['batch_size']
    m = kwargs['steps']
    scheduler = kwargs.get('lr', None)
    opt = kwargs['opt']
    eps = kwargs.get('eps', 2 / 255)
    validate = test_dataset is not None
    metrics = kwargs.get('metrics', paddle.metric.Accuracy())
    eval_freq = kwargs.get('eval_freq', 5)
    loss_fn = paddle.nn.CrossEntropyLoss()
    rank = dist.get_rank()
    nprocs = dist.get_world_size()

    batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True,
                                                      drop_last=True)
    train_loader = paddle.io.DataLoader(train_dataset, batch_sampler=batch_sampler)
    if validate:
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)

    # resume training
    resume_epoch = -1
    if kwargs['weights'] is not None:
        resume_epoch = resume_weights(model=model,
                                      model_path=kwargs['weights'],
                                      optimizer=opt,
                                      opt_path=kwargs['opt_weights'])

    logger.info('start training ... ')
    train_time = 0
    for epoch in range(resume_epoch + 1, epoch_num):
        epoch_time = time.time()
        # turn model into training mode
        model.train()
        # make sure gradient flow for model parameter
        for param in model.parameters():
            param.stop_gradient = False

        total_loss = 0
        total_acc = 0

        for batch_id, data in enumerate(train_loader):
            x, y = data
            y = paddle.unsqueeze(y, axis=1)
            for i in range(m):
                x.stop_gradient = False
                outs = model(x)
                loss = loss_fn(outs, y)
                loss.backward()

                # perturb input image
                imgs_grad = x.grad
                eta = paddle.sign(imgs_grad) * eps
                eta = paddle.clip(eta, -eps, eps)
                x.stop_gradient = True
                x += eta

                # update model parameters
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                opt.clear_grad()

                del eta, imgs_grad
                gc.collect()
            acc = paddle.metric.accuracy(outs, y)
            total_loss += loss
            total_acc += acc
        if nprocs < 2 or rank == 0:
            cur_time = time.time()
            epoch_cost = cur_time - epoch_time
            train_time += epoch_cost
            msg = "Epoch {}, Loss: {}, Acc: {}, Epoch Cost: {:4f}s" \
                .format(epoch, total_loss.numpy() / len(train_loader), total_acc.numpy() / len(train_loader),
                        epoch_cost)
            logger.info(msg)

        # evaluate model after one epoch
        if rank == 0 and validate and np.mod(epoch + 1, eval_freq) == 0:
            model.eval()
            with paddle.no_grad():
                for data in test_loader:
                    x, y = data
                    y = paddle.unsqueeze(y, axis=1)
                    preds = model(x)
                    correct = metrics.compute(preds, y)
                    metrics.update(correct)
                res = metrics.accumulate()

                if nprocs < 2 or rank == 0:
                    msg = "[Validation] Epoch {}, Acc: {}".format(epoch, res)
                    logger.info(msg)
            metrics.reset()

        model.train()

        # save the model at the end of each epoch
        if rank == 0 and np.mod(epoch, 10) == 0:
            save_model(model, opt, save_path, 'epoch'.format(epoch), epoch)

    total_time = time.time() - train_time
    logger.info('Training Finished, Exiting ... ')
    logger.info('Total training time {}h {}m {}s'.format(total_time // 3600,
                                                         np.mod(total_time, 3600) // 60,
                                                         np.mod(np.mod(total_time, 3600), 60)))
    if rank == 0:
        save_model(model, opt, save_path, 'last_model', epoch)
    if dist.get_world_size() > 1:
        paddle.distributed.barrier()
    return


def run(args):
    # Load dataset
    if args.dataset == 'cifar10':
        transforms = T.Compose([T.Resize([224, 224]),
                                T.Transpose(),
                                T.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
                                T.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
        train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transforms)
        test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transforms)

        # Initialize parallel environment
        init_para_env()
        m = load_model(args.model, num_classes=10)
    else:
        from examples.image_cls.miniimagenet import MINIIMAGENET

        transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset_path = os.path.join(os.path.realpath(__file__ + "../" * 3),
                                          'examples/dataset/mini-imagenet/mini-imagenet-cache-train.pkl')
        test_dataset_path = os.path.join(os.path.realpath(__file__ + "../" * 3),
                                         'examples/dataset/mini-imagenet/mini-imagenet-cache-test.pkl')
        label_path = os.path.join(os.path.realpath(__file__ + "../" * 3),
                                  'examples/dataset/mini-imagenet/mini_imagenet_labels.txt')

        train_dataset = MINIIMAGENET(dataset_path=train_dataset_path,
                                     label_path=label_path,
                                     mode='train',
                                     transform=transform)
        test_dataset = MINIIMAGENET(dataset_path=test_dataset_path,
                                    label_path=label_path,
                                    mode='test',
                                    transform=transform)
        init_para_env()
        m = load_model(args.model, num_classes=100)
    
    # Load model
    if dist.get_world_size() > 1:
        m = paddle.DataParallel(m)
    m.train()

    BATCH_SIZE = args.batch_size
    scheduler = None
    num_per_epoch = len(train_dataset) // (dist.get_world_size() * BATCH_SIZE)
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr,
                                                                 T_max=args.epoch * num_per_epoch)
        elif args.scheduler == 'piecewise':
            bounds = [int(num_per_epoch * args.epoch * 0.95)]
            val = [args.lr, args.lr * 0.1]
            scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=bounds, values=val)
        else:
            scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=args.lr, factor=0.2,
                                                            patience=3 * num_per_epoch)
        lr = scheduler
    else:
        lr = args.lr

    if args.warmup:
        scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=lr,
                                                     warmup_steps=num_per_epoch,
                                                     start_lr=0.0001,
                                                     end_lr=args.lr)
        lr = scheduler

    regularizer = None
    if args.regularizer is not None:
        if args.regularizer == 'l1':
            regularizer = paddle.regularizer.L1Decay(1e-4)
        else:
            regularizer = paddle.regularizer.L2Decay(1e-4)

    opt_config = {'learning_rate': lr,
                  'parameters': m.parameters(),
                  'weight_decay': regularizer}

    if args.opt == 'momentum':
        opt = paddle.optimizer.Momentum(**opt_config)
    elif args.opt == 'rmsprop':
        opt = paddle.optimizer.RMSProp(**opt_config)
    elif args.opt == 'adam':
        opt_config.pop('weight_decay')
        opt = paddle.optimizer.Adam(**opt_config)

    metrics = paddle.metric.Accuracy()

    kwargs = {'epoch_num': args.epoch,
              'advtrain_start_num': 0,
              'steps': args.steps,
              'batch_size': BATCH_SIZE,
              'opt': opt,
              'metrics': metrics,
              'weights': args.weights,
              'lr': scheduler,
              'opt_weights': args.opt_weights}

    save_path = os.path.join(os.path.dirname(__file__), args.save_path)
    free_advtrain(m, train_dataset, test_dataset, save_path, **kwargs)


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    run(args)
