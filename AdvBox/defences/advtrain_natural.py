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
paddle2 model natural adversarial training demo.
* implemented random sampling adversarial augmentation.
"""
import sys
import os
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
import paddle.distributed as dist
import numpy as np
from defences.pgd_perturb import PGDTransform
from defences.utils import *
import time

sys.path.append("../..")


def adversarial_train_natural(model, train_set, test_set, save_path=None, **kwargs):
    """
    A demo for adversarial training based on data augmentation.
    Args:
        model: paddle model.
        train_set: paddle dataloader.
        test_set: paddle dataloader.
        save_path: str. path for saving model.
        **kwargs: Other named arguments.
    Returns:
        training log
    """
    assert save_path is not None
    print('start training ... ')
    epoch_num = kwargs["epoch_num"]
    advtrain_start_num = kwargs["advtrain_start_num"]
    batch_size = kwargs["batch_size"]
    adversarial_trans = kwargs["adversarial_trans"]
    opt = kwargs["optimizer"]
    lr = kwargs['lr']
    validate = test_set is not None
    metrics = kwargs.get('metrics', paddle.metric.Accuracy())
    eval_freq = 5
    main_proc = dist.get_world_size() < 2 or dist.get_rank() == 0
    start_time = time.time()
    logger = setup_logger('advtrain')

    # resume training
    resume_epoch = -1
    if kwargs['weights'] is not None:
        resume_epoch = resume_weights(model=model,
                                      model_path=kwargs['weights'],
                                      optimizer=opt,
                                      opt_path=kwargs['opt_weights'])
        print(resume_epoch)

    batch_sampler = paddle.io.DistributedBatchSampler(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader = paddle.io.DataLoader(train_set, batch_sampler=batch_sampler)
    if validate:
        test_loader = paddle.io.DataLoader(test_set, batch_size=batch_size)
    train_time = time.time()
    for epoch in range(resume_epoch + 1, epoch_num):
        losses = 0
        accuracies = 0
        # make sure gradient flow for model parameter
        for param in model.parameters():
            param.stop_gradient = False

        epoch_time = time.time()
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.unsqueeze(data[1], 1)

            # adversarial training late start
            if epoch >= advtrain_start_num and adversarial_trans is not None:
                with model.no_sync():
                    x_data_augmented = adversarial_trans(x_data, y_data)
                    model.clear_gradients()
            else:
                x_data_augmented = x_data

            # turn model into training mode
            model.train()

            logits = model(x_data_augmented)
            loss = F.cross_entropy(logits, y_data)
            loss.backward()
            opt.step()
            if lr is not None:
                lr.step()
            opt.clear_grad()

            losses += loss
            acc = paddle.metric.accuracy(logits, y_data).numpy()
            accuracies += acc
        epoch_cost = time.time() - epoch_time
        if main_proc:
            logger.info("Epoch {}, Loss {}, Acc {}, Cost {}m {}s".format(epoch, losses / len(train_loader),
                                                                         accuracies / len(train_loader),
                                                                         epoch_cost // 60,
                                                                         np.mod(epoch_cost, 60)))

        # evaluate model after one epoch
        if main_proc and validate and np.mod(epoch + 1, eval_freq) == 0:
            model.eval()
            losses = 0
            with paddle.no_grad():
                for batch_id, data in enumerate(test_loader()):
                    x_data = data[0]
                    y_data = paddle.unsqueeze(data[1], 1)

                    logits = model(x_data)
                    loss = F.cross_entropy(logits, y_data)
                    correct = metrics.compute(logits, y_data)
                    metrics.update(correct)
                    losses += loss
            acc = metrics.accumulate()
            logger.info('[Validation] Epoch {}, Acc {}, Loss {}'.format(epoch, acc, losses / len(test_loader)))

        if main_proc and np.mod(epoch + 1, 10) == 0:
            save_model(model, opt, save_path, 'epoch{}'.format(epoch), epoch)

        model.train()

    total_time = time.time() - train_time
    logger.info('Training Finished, Exiting ... ')
    logger.info('Total training time {}h {}m {}s'.format(total_time // 3600,
                                                         np.mod(total_time, 3600) // 60,
                                                         np.mod(np.mod(total_time, 3600), 60)))
    if main_proc:
        save_model(model, opt, save_path, 'last_model', epoch)
    if dist.get_world_size() > 1:
        paddle.distributed.barrier()


def run(args):
    if args.dataset == 'cifar10':
        num_classes = 10
        MEAN = [0.491, 0.482, 0.447]
        STD = [0.247, 0.243, 0.262]
        transforms = T.Compose([T.Resize([224, 224]),
                                T.Transpose(),
                                T.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
                                T.Normalize(mean=MEAN, std=STD)])
        train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transforms)
        test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transforms)
    else:
        from examples.image_cls.miniimagenet import MINIIMAGENET
        num_classes = 100
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        transform = T.Compose([T.Normalize(MEAN, STD, data_format='CHW')])
        train_dataset_path = os.path.join(os.path.realpath(__file__ + "../" * 3),
                                          'examples/dataset/mini-imagenet/mini-imagenet-cache-train.pkl')
        test_dataset_path = os.path.join(os.path.realpath(__file__ + "../" * 3),
                                         'examples/dataset/mini-imagenet/mini-imagenet-cache-test.pkl')
        label_path = os.path.join(os.path.realpath(__file__ + "../" * 3),
                                  'examples/dataset/mini-imagenet/mini_imagenet_labels.txt')

        test_dataset = MINIIMAGENET(dataset_path=test_dataset_path,
                                    label_path=label_path,
                                    mode='test',
                                    transform=transform)
        train_dataset = MINIIMAGENET(dataset_path=train_dataset_path,
                                     label_path=label_path,
                                     mode='train',
                                     transform=transform)

    # Initialize parallel environment
    init_para_env()
    # Load model
    m = load_model(args.model, num_classes=num_classes)
    m.eval()
    m = paddle.DataParallel(m)

    batch_size = args.batch_size
    attack_config = {'num_classes': num_classes, 'model_mean': MEAN, 'model_std': STD}
    advtrans = PGDTransform(m, attack_config, p=args.attack_prob)
    scheduler = None
    num_per_epoch = len(train_dataset) // (dist.get_world_size() * batch_size)
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.epoch)
        elif args.scheduler == 'piecewise':
            bounds = [int(num_per_epoch * args.epoch * 0.9)]
            val = [args.lr, args.lr * 0.1]
            scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=bounds, values=val)
        else:
            scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=args.lr, factor=0.2, threshold=1e-2,
                                                            patience=num_per_epoch)
        lr = scheduler
    else:
        lr = args.lr

    if args.warmup:
        scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=lr,
                                                     warmup_steps=int(num_per_epoch),
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

    kwargs = {'epoch_num': args.epoch,
              'advtrain_start_num': 0,
              'batch_size': batch_size,
              'adversarial_trans': advtrans,
              'optimizer': opt,
              'weights': args.weights,
              'lr': scheduler,
              'opt_weights': args.opt_weights}

    save_path = os.path.join(os.path.dirname(__file__), args.save_path)
    adversarial_train_natural(model=m,
                              train_set=train_dataset,
                              test_set=test_dataset,
                              save_path=save_path,
                              **kwargs)


if __name__ == '__main__':
    args = parse_args()
    run(args)
