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
utility tools.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import distutils.util
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time

OUTPUT = './output/'
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

class bcolors:
    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    CYAN = "\033[1;36m"
    GREEN = "\033[1;32m"
    RESET = "\033[0;0m"
    BOLD = "\033[;1m"
    REVERSE = "\033[;7m"
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_arguments(args):
    """Print argparse's arguments.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)
    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def check_output_directory(type):
    """
    create output directory
    Args:
         type: name of picture set for test
    """
    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT, 0o755)
    if not os.path.exists(OUTPUT + "/" + type):
        os.mkdir(OUTPUT + "/" + type, 0o755)


def convert_net(img_example):
    """
    convert image array to original
    Args:
         img_example: array data of img
    """
    #reshape img_example
    output_img = np.reshape(img_example.astype('float32'), (3, 224, 224))

    output_img *= img_std
    output_img += img_mean
    output_img *= 255
    output_img = np.reshape(output_img.astype(np.uint8), (3, 224, 224))

    #convert C,H,W to H,W,C
    output_img = output_img.transpose((1, 2, 0))

    return output_img


def save_image(output_img, path):
    """
    save image from array that original or adversarial
    Args:
         img_example: array data of img
         path: directory and filename
    """
    im = Image.fromarray(output_img)
    im.save(path, 'png')


def generation_image(id, org_img, org_label, adv_img, adv_label, attack_method='FGSM'):
    """
    save image from array that original or adversarial
    imagenet data set
    Args:
         org_img: array data of test img
         adv_img: array data of adv img
         org_label: the inference label of test image
         adv_label: the adverarial label of adv image
         attack_method: the adverarial example generation method
    """
    DATA_TYPE = "imagenet"
    check_output_directory(DATA_TYPE)

    org_path= OUTPUT + DATA_TYPE + "/%d_original-%d-by-%s.png" \
              % (id, org_label, attack_method)
    adv_path= OUTPUT + DATA_TYPE + "/%d_adversary-%d-by-%s.png" \
              % (id, adv_label, attack_method)
    diff_path= OUTPUT + DATA_TYPE + "/%d_diff-x-by-%s.png" % (id, attack_method)

    org_output = convert_net(org_img)
    adv_output = convert_net(adv_img)
    diff_output = abs(adv_output - org_output)

    save_image(org_output, org_path)
    save_image(adv_output, adv_path)
    save_image(diff_output, diff_path)
    print("--------------------------------------------------")


def show_images_diff(original_img, original_label, adversarial_img, adversarial_label):
    """
    show original image, adversarial image and their difference
    Args:
        original_img: original image, numpy
        original_label:original label, int 
        adversarial_img: adversarial image
        adversarial_label: adversarial label

    Returns:

    """

    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img

    l0 = np.where(difference != 0)[0].shape[0]
    l2 = np.linalg.norm(difference)
    print("l0={} l2={}".format(l0, l2))

    #(-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    ts = time.localtime(time.time())
    ts = time.strftime("%Y-%m-%d %H:%M:%S", ts)

    if not os.path.exists('output'):
        os.makedirs('output')
    plt.savefig("output/orig_adv_diff_{}_{}.png".format(adversarial_label, ts))
    plt.show()

def show_images_diff_denoising(image_a, image_a_label, image_b, image_b_label, image_a_title='Input', image_b_title='output'):
    """
    show original image, adversarial image and their difference
    Args:
        image_a: original image, ndarray
        image_a_label:original label, int
        image_b: adversarial image, ndarray
        image_b_label: adversarial label
        image_a_title: the title of the image a
        image_b_title: the title of the image b

    Returns:

    """

    plt.figure()

    plt.subplot(131)
    plt.title(image_a_title)
    plt.imshow(image_a)
    plt.axis('off')

    plt.subplot(132)
    plt.title(image_b_title)
    plt.imshow(image_b)
    plt.axis('off')

    plt.subplot(133)
    plt.title(image_a_title+'-'+image_b_title)
    difference = image_a - image_b

    l0 = np.where(difference != 0)[0].shape[0]
    l2 = np.linalg.norm(difference)
    print("l0={} l2={}".format(l0, l2))

    #(-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    ts = time.localtime(time.time())
    ts = time.strftime("%Y-%m-%d %H:%M:%S", ts)

    if not os.path.exists('examples/image_cls/output'):
        os.makedirs('output')
    plt.savefig("output/{}_{}_diff_{}_{}_{}.png".format(image_a_title, image_b_title, image_a_label, image_b_label, ts))
    plt.show()


def show_input_adv_and_denoise(image_a, image_b, image_c, image_d, \
                               image_a_label, image_b_label, image_c_label, image_d_label, \
                               image_a_title='Input', image_b_title='Adversary', \
                               image_c_title='Adv-Denoise', image_d_title='In-Denoise',method='Default'
):
    """
    show original image, adversarial image, and their denoising results, respectively
    Args:
        image_a: original image, ndarray
        image_a_label: original label, str
        image_a_title: the title of the image a
        image_b: adversarial image, ndarray
        image_b_label: adversarial label
        image_b_title: the title of the image b
        image_c: denoising result of the adversarial image, ndarray
        image_c_label: the predicted class label after denoising of the adv-image
        image_c_title: the title of the image c
        image_d: denoising result of the original input image, ndarray
        image_d_label: the predicted class label after denoising of the input image
        image_d_title: the title of the image d

    Returns:

    """
    # get the first class name
    a_label=''
    for i in image_a_label:
        if i!=',':
            a_label+=i
        else:
            break
    temp=a_label
    if len(a_label)>10:
        temp=''
        for i in a_label:
            if i==' ':
                temp=''
            else:
                temp=temp+i
    a_label=temp
    b_label=''
    for i in image_b_label:
        if i!=',':
            b_label+=i
        else:
            break
    temp=b_label
    if len(b_label)>10:
        temp=''
        for i in b_label:
            if i==' ':
                temp=''
            else:
                temp=temp+i
    b_label=temp
    c_label=''
    for i in image_c_label:
        if i!=',':
            c_label+=i
        else:
            break
    temp=c_label
    if len(c_label)>10:
        temp=''
        for i in c_label:
            if i==' ':
                temp=''
            else:
                temp=temp+i
    c_label=temp
    d_label=''
    for i in image_d_label:
        if i!=',':
            d_label+=i
        else:
            break
    temp=d_label
    if len(d_label)>10:
        temp=''
        for i in d_label:
            if i==' ':
                temp=''
            else:
                temp=temp+i
    d_label=temp

    # define the plot position
    w = image_c.shape[0] if image_c.shape[0] > image_d.shape[0] else image_d.shape[0]
    h = image_c.shape[1] if image_c.shape[1] > image_d.shape[1] else image_d.shape[1]
    x = 0      # initial horizontal position of the first line
    y = h + 10    # initial vertical position of the first line
    xos = 15    # offset to x of the second line
    yos = 10    # offset to y of the second line


    fig = plt.figure()

    title = 'Denoise method: ' + method
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.80)

    plt.subplot(141)
    plt.title(image_a_title)
    plt.imshow(image_a)
    plt.text(x, y, 'Top1 label:')
    plt.text(x+xos, y+yos, a_label)
    plt.axis('off')

    plt.subplot(142)
    plt.title(image_b_title)
    plt.imshow(image_b)
    plt.text(x, y, 'Top1 label:')
    plt.text(x+xos, y+yos, b_label)
    plt.axis('off')

    plt.subplot(143)
    plt.title(image_c_title)
    plt.imshow(image_c)
    plt.text(x, y, 'Top1 label:')
    plt.text(x+xos, y+yos, c_label)
    plt.axis('off')

    plt.subplot(144)
    plt.title(image_d_title)
    plt.imshow(image_d)
    plt.text(x, y, 'Top1 label:')
    plt.text(x+xos, y+yos, d_label)
    plt.axis('off')

    plt.tight_layout()

    if not os.path.exists('examples/image_cls/output'):
        os.makedirs('output')
    plt.savefig("output/{}_Denoising_Comparison.png".format(method))
    plt.show()


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
