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
""" Test case for Paddle """
from __future__ import absolute_import

import paddle
import numpy as np
from perceptron.utils.image import load_image
from perceptron.benchmarks.brightness import BrightnessMetric
from perceptron.utils.criteria.classification import Misclassification
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import bcolors
from perceptron.models.classification.paddlemodelupload import PaModelUpload
import os

here = os.path.dirname(os.path.abspath(__file__))

# interpret the label as human language
with open(os.path.join(here, 'User_Model/labels.txt')) as info:
    image_dict = eval(info.read())

is_available = len(paddle.static.cuda_places()) > 0
print('use gpu: ', is_available)

# initialize the PaddleModel
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
# !! Here 10 is to be changed to the number of predicted categories of the user model
fmodel = PaModelUpload(
    bounds=(0, 1), num_classes=10, preprocessing=(mean, std))

# get source image and label
image = load_image(shape=(32, 32), data_format='channels_first',
                   path=os.path.join(os.path.dirname(__file__),
                                     '../perceptron/utils/images/%s' % 'cifar0.png'))  # specify the attack image

# set the type of noise which will used to generate the adversarial examples
metric = BrightnessMetric(fmodel, criterion=Misclassification())

# set the label as the predicted one
label = np.argmax(fmodel.predictions(image))

print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
# set 'unpack' as false so we can access the detailed info of adversary
adversary = metric(image, label, verify=False, unpack=False)
print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

if adversary.image is None:
    print(
        bcolors.WARNING +
        'Warning: Cannot find an adversary!' +
        bcolors.ENDC)
    exit(-1)

###################  print summary info  #####################################

keywords = ['Paddle', 'UserUploadModel', 'Misclassification', 'Brightness']

true_label = np.argmax(fmodel.predictions(image))
fake_label = np.argmax(fmodel.predictions(adversary.image))

print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
print('Configuration:' + bcolors.CYAN + ' --framework %s '
                                        '--model %s --criterion %s '
                                        '--metric %s' % tuple(keywords) + bcolors.ENDC)
print('The predicted label of original image is '
      + bcolors.GREEN + image_dict[true_label] + bcolors.ENDC)
print('The predicted label of adversary image is '
      + bcolors.RED + image_dict[fake_label] + bcolors.ENDC)
print('Minimum perturbation required: %s' % bcolors.BLUE
      + str(adversary.distance) + bcolors.ENDC)
print('Verifiable bound: %s' % bcolors.BLUE
      + str(adversary.verifiable_bounds) + bcolors.ENDC)
print('\n')

plot_image(adversary,
           title=', '.join(keywords),
           figname=os.path.join(here, 'images/%s.png' % '_'.join(keywords)))
