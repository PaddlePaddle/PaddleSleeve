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
* implemented Free-AT
"""
import os
import sys
sys.path.append("../..")

import paddle
import numpy as np
import gc
from PIL import Image
from obj_detection.attack.utils.tools import denormalize_image

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def save_model(model, opt, save_dir, save_name, last_epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    paddle.save(model.state_dict(), save_path + ".pdparams")
    state_dict = opt.state_dict()
    state_dict['last_epoch'] = last_epoch
    paddle.save(state_dict, save_path + ".pdopt")


def free_advtrain(model, train_loader, test_loader=None, save_path=None, **kwargs):
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
    val_acc_history = []
    val_loss_history = []
    epoch_num = kwargs["epoch_num"]
    m = kwargs['steps']
    assert np.mod(epoch_num, m) == 0, "epoch_num should be divisible by num_steps"
    scheduler = kwargs['scheduler']
    opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    collate_batch = kwargs['collate_batch']
    eps = kwargs.get('eps', 4 / 255)
    validate = test_loader is not None
    metrics = kwargs['metrics'] if validate else None
    best_ap = 0

    print('start training ... ')
    for epoch in range(epoch_num // m):
        print("epoch: {}".format(epoch))
        # turn model into training mode
        model.train()
        # make sure gradient flow for model parameter
        for param in model.parameters():
            param.stop_gradient = False

        for batch_id, data in enumerate(train_loader):
            if collate_batch:
                data = collate_batch(data)

            for i in range(m):
                data['image'].stop_gradient = False
                outs = model(data)
                loss = outs['loss']
                loss.backward()

                # perturb input image
                imgs_grad = data['image'].grad
                eta = paddle.sign(imgs_grad) * eps
                eta = paddle.clip(eta, -eps, eps)
                data['image'].stop_gradient = True
                data['image'] += eta

                # update model parameters
                opt.step()
                scheduler.step()
                opt.clear_grad()
                del outs, eta, imgs_grad
                gc.collect()

        # evaluate model after one epoch
        if validate:
            assert metrics is not None, 'Please provide a valid metric'
            model.eval()
            with paddle.no_grad():
                sample_num = 0
                for step_id, data in enumerate(test_loader):
                    # forward
                    outs = model.get_pred(data)

                    # update metrics
                    for metric in metrics:
                        metric.update(data, outs)

                    sample_num += data['im_id'].numpy().shape[0]

                # accumulate metric to log out
                for metric in metrics:
                    metric.accumulate()
                    metric.log()
                # save the model if it has better performance
                best_model = None
                save_name = None
                for metric in metrics:
                    map_res = metric.get_results()
                    if 'bbox' in map_res:
                        key = 'bbox'
                    elif 'keypoint' in map_res:
                        key = 'keypoint'
                    else:
                        key = 'mask'
                    if map_res[key][0] > best_ap:
                        best_ap = map_res[key][0]
                        save_name = 'best_model'
                        best_model = model

                if best_model:
                    save_model(model, opt, save_path,
                               save_name, epoch+1)
                # reset metric states for metric may performed multiple times
                for metric in metrics:
                    metric.reset()
        save_model(model, opt, save_path, 'epoch_{}'.format(epoch), epoch)
        model.train()

    # save the model at the end of training
    save_model(model, opt, save_path, 'last_model', epoch_num)

    return val_acc_history, val_loss_history, model, opt

