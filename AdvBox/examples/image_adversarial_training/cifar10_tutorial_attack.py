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
A tutorial for LD attack adv sample generation on CIFAR10 dataset.
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
from adversary import Adversary
from models.whitebox import PaddleWhiteBoxModel
from skimage.metrics import structural_similarity

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def get_best_weigthts_from_folder(folder, pdparams_file_starter):
    pdparams_files = [filename for filename in os.listdir(folder) if filename.lower().endswith('.pdparams')
                      and filename.lower().startswith(pdparams_file_starter.lower())]
    if not pdparams_files:
        return None
    else:
        acc_list = [filename.split('.')[1] for filename in pdparams_files]
        max_index = acc_list.index(max(acc_list))
        best_weight_path = os.path.join(folder, pdparams_files[max_index])
        print('Loaded: ', best_weight_path)
    return best_weight_path


# Initialize model structure and load trained parameters
from main_setting import MODEL, MODEL_PATH, MODEL_PARA_NAME
MODEL = MODEL
path = get_best_weigthts_from_folder(MODEL_PATH, MODEL_PARA_NAME)
model_state_dict = paddle.load(path)
MODEL.set_state_dict(model_state_dict)

from main_setting import cifar10_test, MEAN, STD
MEAN = MEAN
STD = STD
CIFAR10_TEST = cifar10_test

attack_zoo = ("FGSM", "LD")
attack_choice = input(f"choose {attack_zoo}:")
assert attack_choice in attack_zoo

if attack_choice == attack_zoo[0]:
    from attacks.gradient_method import FGSM
    ATTACK_METHOD = FGSM
    INIT_CONFIG = {"norm": "Linf", "epsilon_ball": 8/255}
    ATTACK_CONFIG = {"norm_ord": np.inf, "epsilons": 0.003, "epsilon_steps": 1, "steps": 1}
elif attack_choice == attack_zoo[1]:
    from attacks.logits_dispersion import LD
    ATTACK_METHOD = LD
    INIT_CONFIG = {"norm": "Linf", "epsilon_ball": 8/255, "dispersion_type": "softmax_kl"}
    ATTACK_CONFIG = {"perturb_steps": 10, "verbose": True}
else:
    exit(1)

TOTAL_TEST_NUM = 500
# for now, attacks only support batch == 1, thus we fixed batch size.
BATCH_SIZE = 1
IS_TARGET_ATTACK = False


def main(model, advbox_model, test_loader, attack, attack_config=None):
    """
    Demonstrates how to use attacks.
    """
    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    correct_num = 0
    pbar = tqdm(total=len(test_loader()))

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

        # it is a must or BN will change during forwarding
        model.eval()
        logits = model(data[0])
        acc = paddle.metric.accuracy(logits, data[1].unsqueeze(0))
        if int(acc) == 1:
            correct_num += 1
        else:
            pass

        pbar.update(1)
        if adversary.is_successful():
            if not USE_GPU.startswith('gpu'):
                show_images_diff(adversary.original, adversary.original_label,
                                 adversary.adversarial_example.squeeze(), adversary.adversarial_label)
            fooling_count += 1
            psnr = compute_psnr(adversary.original, adversary.adversarial_example.squeeze())
            ssim = compute_ssim(np.rollaxis(adversary.original, 0, 3),
                                np.rollaxis(adversary.adversarial_example.squeeze(), 0, 3))
            pbar.set_description('succeeded, psnr=%f, ssim=%f, '
                                 'original_label=%d, adversarial_label=%d, count=%d'
                                 % (psnr, ssim, data[1], adversary.adversarial_label, total_count))
        else:
            pbar.set_description('failed, original_label=%d, count=%d'
                                 % (data[1], total_count))

        if total_count >= TOTAL_TEST_NUM:
            print("[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f, model_acc=%f"
                  % (fooling_count, total_count, float(fooling_count) / total_count, correct_num / total_count))
            break

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
    if original_img.any() > 1.0:
        original_img = original_img / 255
    if adversarial_img.any() > 1.0:
        adversarial_img = adversarial_img / 255

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
    # init a paddle model
    advbox_model = PaddleWhiteBoxModel(
        [MODEL],
        [1],
        (0, 1),
        mean=MEAN,
        std=STD,
        input_channel_axis=0,
        input_shape=(3, 256, 256),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=10)

    test_loader = paddle.io.DataLoader(cifar10_test, batch_size=BATCH_SIZE)

    attack = ATTACK_METHOD(advbox_model, **INIT_CONFIG)

    main(MODEL, advbox_model, test_loader, attack, ATTACK_CONFIG)
