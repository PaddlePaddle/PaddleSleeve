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
A tutorial for model evaluation on dataset.
"""
from __future__ import print_function
from __future__ import absolute_import

import paddle
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import math
import sys
sys.path.append("../..")
import argparse
import functools

from adversary import Adversary
from models.whitebox import PaddleWhiteBoxModel
from skimage.metrics import structural_similarity
from examples.utils import add_arguments, print_arguments, get_best_weigthts_from_folder
from main_setting import model_zoo, training_zoo, dataset_zoo, attack_zoo, get_model_setting, get_save_path, get_train_method_setting, get_dataset, get_attack_setting, assert_input, get_model_para_name

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model', str, 'preactresnet', 'The model to evaluate, choose {model_zoo}.'.format(model_zoo=model_zoo))
add_arg('training_method', str, 'base', 'The training method of the model to be evaluated, choose in {training_zoo}.'.format(training_zoo=training_zoo))
add_arg('attack_method', str, 'FGSM', 'The attack method of the model to be evaluated, choose in {attack_zoo}. Only works if the "training_method" is not "base"'.format(attack_zoo=attack_zoo))
add_arg('dataset', str, 'cifar10', 'The training dataset for the model to be evaluated, choose in {dataset_zoo}.'.format(dataset_zoo=dataset_zoo))
add_arg('use_base_pretrain', str, 'no', 'Whether the model to be evaluated uses a pre-trained model trained in base mode, choose in ("yes", "no").')


USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


# TOTAL_TEST_NUM = 500
IS_TARGET_ATTACK = False


def main():
    """
    Demonstrates how to use attacks.
    """
    
    args = parser.parse_args()
    print_arguments(args)

    model_choice = args.model
    training_choice = args.training_method
    dataset_choice = args.dataset
    attack_choice = args.attack_method
    use_base_pretrain = args.use_base_pretrain

    assert_input(model_choice, training_choice, dataset_choice, attack_choice, use_base_pretrain)

    model, MEAN, STD, _, test_transform = get_model_setting(model_choice, dataset_choice)

    # Initialize model structure and load trained parameters
    model_dir = get_save_path(model_choice, training_choice, dataset_choice, attack_choice, use_base_pretrain)
    model_para_name = get_model_para_name(training_choice)
    model_path = get_best_weigthts_from_folder(model_dir, model_para_name)
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)

    attack_method, init_config, attack_config = get_attack_setting(attack_choice)

    adverarial_train, enhance_config, advtrain_settings = get_train_method_setting(model, training_choice)
    
    test_set, class_num = get_dataset(dataset_choice, 'test', test_transform)

    # for now, attacks only support batch == 1, thus we fixed batch size.
    batch_size = 1
    test_loader = paddle.io.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    data = test_loader().next()
    

    # init a paddle model
    advbox_model = PaddleWhiteBoxModel(
        [model],
        [1],
        (0, 1),
        mean=MEAN,
        std=STD,
        input_channel_axis=0,
        input_shape=tuple(data[0].shape[1:]),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=class_num)

    attack = attack_method(advbox_model, **init_config)


    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    correct_num = 0
    pbar = tqdm(total=len(test_loader()))

    model.eval()
    for data in test_loader():
        total_count += 1
        img = data[0][0]
        label = data[1]

        if IS_TARGET_ATTACK:
            # init adversary status
            adversary = Adversary(img.numpy(), int(label))
            target = np.random.randint(advbox_model.num_classes())
            while label == target:
                target = np.random.randint(advbox_model.num_classes())
            adversary.set_status(is_targeted_attack=True, target_label=target)
            # run call to attack, change adversary's status
            adversary = attack(adversary)
        else:
            adversary = Adversary(img.numpy(), int(label))
            # run call to attack, change adversary's status
            adversary = attack(adversary, **attack_config)

        logits = model(data[0])
        acc = paddle.metric.accuracy(logits, data[1].unsqueeze(0))
        if int(acc) == 1:
            correct_num += 1
        else:
            pass

        pbar.update(1)
        if adversary.is_successful():
            if not USE_GPU.startswith('gpu'):
                show_images_diff(adversary.denormalized_original, adversary.original_label,
                                 adversary.denormalized_adversarial_example, adversary.adversarial_label)
            fooling_count += 1
            psnr = compute_psnr(adversary.denormalized_original, adversary.denormalized_adversarial_example)
            ssim = compute_ssim(np.rollaxis(adversary.denormalized_original, 0, 3),
                                np.rollaxis(adversary.denormalized_adversarial_example, 0, 3))
            pbar.set_description('succeeded, psnr=%f, ssim=%f, '
                                 'original_label=%d, adversarial_label=%d, count=%d'
                                 % (psnr, ssim, data[1], adversary.adversarial_label, total_count))
        else:
            pbar.set_description('failed, original_label=%d, count=%d'
                                 % (data[1], total_count))

        # if total_count >= TOTAL_TEST_NUM:
        #     print("[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f, model_acc=%f"
        #           % (fooling_count, total_count, float(fooling_count) / total_count, correct_num / total_count))
        #     break
    print("[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f, model_acc=%f" % (fooling_count, total_count, float(fooling_count) / total_count, correct_num / total_count))

    print("Attack done")


def compute_psnr(img1, img2):
    """
    Peak signal-to-noise ratio (PSNR)
    Args:
        img1: a numpy.array of image 1
        img2: a numpy.array of image 2
    Returns:
        PSNR ratio
    """
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    pixel_max = 1
    return 20 * math.log10(pixel_max / math.sqrt(mse))


def compute_ssim(im1, im2):
    """
    Structural SIMilarity (SSIM)
    Args:
        im1: a numpy.array of image 1
        im2: a numpy.array of image 2
    Returns:
        SSIM value
    """
    ssim_value = structural_similarity(im1 * 255, im2 * 255, data_range=255, multichannel=True)
    return ssim_value


def show_images_diff(original_img, original_label, adversarial_img, adversarial_label, outdir='pic-output'):
    """
    Compare original and adv perturbed images.
    Args:
        original_img: original image
        original_label: original image label
        adversarial_img: perturbed image
        adversarial_label: perturbed image label
    Returns:
        None
    """
    if original_img.shape[0] == 3:
        original_img = np.rollaxis(original_img, 0, 3)

    if adversarial_img.shape[0] == 3:
        adversarial_img = np.rollaxis(adversarial_img, 0, 3)

    plt.figure()
    plt.subplot(131)
    plt.title('Original')
    # plt.imshow(original_img.astype('uint8'))
    # plt.imshow(original_img.astype('float32'))
    plt.imshow(original_img)
    plt.axis('off')
    plt.subplot(132)
    plt.title('Adversarial')
    # plt.imshow(adversarial_img.astype('uint8'))
    # plt.imshow(adversarial_img.astype('float32'))
    plt.imshow(adversarial_img)
    plt.axis('off')
    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img
    print ("diff shape: ", difference.shape)
    # (-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    # plt.imshow(difference.astype('uint8'), cmap=plt.cm.gray)
    # plt.imshow(difference.astype('float32'), cmap=plt.cm.gray)
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig("{}/fgsm_attack_{}.png".format(outdir, adversarial_label))
    plt.show()


if __name__ == '__main__':

    main()
