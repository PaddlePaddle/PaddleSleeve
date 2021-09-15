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

"""Laucher for evaluation tasks."""

from __future__ import absolute_import

import os
import sys

work_dir = os.path.abspath(os.getcwd())
sys.path.append(work_dir)

from perceptron.utils.tools import get_image
from perceptron.utils.tools import get_metric
from perceptron.utils.tools import get_distance
from perceptron.utils.tools import get_criteria
from perceptron.utils.tools import get_model
from perceptron.utils.tools import plot_image
from perceptron.utils.tools import plot_image_objectdetection, plot_image_objectdetection_hub
from perceptron.utils.tools import bcolors
import argparse
import numpy as np

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def validate_cmd(args):
    import json
    import os
    with open(os.path.dirname(os.path.realpath(__file__)) +
              '/utils/summary.json') as f:
        summary = json.load(f)

    # step 1. check the framework
    if args.framework not in summary['frameworks']:
        print(bcolors.RED + 'Please provide a valid framework: ' + bcolors.ENDC
              + ', '.join(summary['frameworks']))
        exit(-1)

    # step 2. check the model
    valid_models = summary[args.framework + '_models']
    if args.model not in valid_models:
        print(bcolors.RED + 'Please provide a valid model implemented under framework '
              + args.framework + ', i.e.: ' + bcolors.ENDC + ', '.join(valid_models))
        exit(-1)

    # step 3. check the criterion
    valid_detection_models = []
    if (args.framework == 'keras' and args.model == 'ssd300') or args.framework == 'paddlehub' or args.framework == 'pytorchhub':
        valid_detection_models = summary['detection_models']
    is_objectdetection = (((args.framework == 'keras' and args.model == 'ssd300')
                           or args.framework == 'paddlehub'
                           or args.framework == 'pytorchhub')
                          and args.model in valid_detection_models)
    if is_objectdetection:
        hint_criterion_string = 'Your should choose criterion that supports object detection model.'
        valid_criterions = summary['detection_criterions']
    elif args.framework == 'cloud':
        hint_criterion_string = 'Your should choose criterion that supports the cloud model.'
        valid_criterions = summary[args.model + '_criterions']
    else:
        hint_criterion_string = 'You should choose criterion that supports classification model.'
        valid_criterions = summary['classification_criterions']
    if args.criteria not in valid_criterions:
        print(bcolors.WARNING + hint_criterion_string + bcolors.ENDC)
        print(bcolors.RED + 'Please provide a valid criterion: ' + bcolors.ENDC
              + ', '.join(valid_criterions))
        exit(-1)

    # step 4. whether the target_class must provided
    must_provide_target = args.criteria in summary['targeted_criteria']
    if args.target_class is None and must_provide_target:
        print(bcolors.RED + 'You are using target-attack! Please provide a target class.' + bcolors.ENDC)
        print('For example: --target_class 282')
        exit(-1)

    # step 5. check the metric
    must_black_box = (args.framework == 'cloud')
    if must_black_box and args.verify is False:
        valid_metrics = summary['blackbox_metrics']
        hint_metric_string = 'Your model only supports black-box attack.'
    elif args.verify is True:
        valid_metrics = summary['verifiable_metrics']
        hint_metric_string = 'You set verify as true, please select a verifiable metric.'
    else:
        hint_metric_string = None
        valid_metrics = summary['metrics']
    if args.metric not in valid_metrics:
        if hint_metric_string:
            print(bcolors.WARNING + hint_metric_string + bcolors.ENDC)
        print(bcolors.RED + 'Please provide a valid metric: ' + bcolors.ENDC +
              ', '.join(valid_metrics))
        exit(-1)


def parse_summary():
    """Parse dictionary from summary json file."""
    import json
    import os
    with open(
            os.path.dirname(os.path.realpath(__file__)) +
            '/utils/summary.json') as f:
        summary = json.load(f)

    summary['models'] = list(set(summary['pytorch_models'] +
                                 summary['keras_models'] +
                                 summary['cloud_models'] +
                                 summary['paddle_models']))
    return summary


def run_attack(args, summary):
    """Run the attack."""
    model = get_model(args.model, args.framework, summary)
    distance = get_distance(args.distance)

    channel_axis = 'channels_first' if (args.framework == 'pytorch' or args.framework == 'paddle') else 'channels_last'
    image_path = os.path.join(os.path.dirname(__file__), 'utils/images/%s' % args.image)
    image = get_image(image_path, args.framework, args.model, channel_axis)

    if args.framework == 'paddlehub':
        predictions = model.predictions(image)
        if 'data' in predictions[0]:  # object detection
            if predictions[0]['data'] == []:
                print("Detection failed.")
                exit(-1)
            label = predictions[0]['data'][0]['label']
        else:  # classification
            label = list(predictions[0].keys())[0]
    elif args.framework == 'pytorchhub':
        predictions = model.predictions(image)
        if len(predictions.pred[0]) == 0:
            print("Detection failed.")
            exit(-1)
        label = predictions.pred[0][0, 5]
    elif args.framework == 'keras':
        predictions = model.predictions(image)
        if args.model == 'ssd300':
            label = predictions['classes'][0]
        else:
            label = np.argmax(predictions)
    elif args.target_class is None and args.framework != 'cloud':
        label = np.argmax(model.predictions(image))
    elif args.framework == 'cloud':
        label = model.predictions(image)
    else:
        label = args.target_class

    if args.framework == 'paddlehub' or \
            args.framework == 'pytorchhub' or \
            args.framework == 'keras':
        criteria = get_criteria(args.criteria, label, predictions, model_name=args.model)
    else:
        criteria = get_criteria(args.criteria, args.target_class)
    metric = get_metric(args.metric, model, criteria, distance)

    print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
    if args.model in ["yolo_v3", "keras_ssd300", "retina_resnet_50"]:
        adversary = metric(image, unpack=False, binary_search_steps=1)
    elif args.metric not in ["carlini_wagner_l2", "carlini_wagner_linf"]:
        if args.metric in summary['verifiable_metrics']:
            adversary = metric(image, label, unpack=False, verify=args.verify)
        else:
            adversary = metric(image, label, unpack=False, epsilons=10)
    else:
        adversary = metric(image, label, unpack=False, binary_search_steps=10, max_iterations=5)
    print(bcolors.BOLD + 'Process finished' + bcolors.ENDC)

    if adversary.image is None:
        print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
        return adversary

    ###################  print summary info  #####################################
    keywords = [args.framework, args.model, args.criteria, args.metric, args.distance]

    # classification models
    if args.model not in summary['detection_models']:
        # interpret the label as human language
        with open('perceptron/utils/labels.txt') as info:
            imagenet_dict = eval(info.read())
        if args.model is 'paddlehub_mobilenet_v2_animals':
            print(args.framework)
            true_label = label
            fake_label = list(adversary.output[0].keys())[0]
        elif args.framework == 'keras':
            print(args.framework)
            true_label = imagenet_dict[label]
            fake_label = imagenet_dict[np.argmax(adversary.output)]
        elif args.framework != 'cloud':
            print(args.framework)
            true_label = imagenet_dict[np.argmax(model.predictions(image))]
            fake_label = imagenet_dict[np.argmax(model.predictions(adversary.image))]
        else:
            print(args.framework)
            true_label = str(model.predictions(image))
            fake_label = str(model.predictions(image))

        print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
        print('Configuration:' + bcolors.CYAN + ' --framework %s '
                                                '--model %s --criterion %s '
                                                '--metric %s --distance %s' % tuple(keywords) + bcolors.ENDC)
        print('The predicted label of original image is '
              + bcolors.GREEN + true_label + bcolors.ENDC)
        print('The predicted label of adversary image is '
              + bcolors.RED + fake_label + bcolors.ENDC)
        print('Minimum perturbation required: %s' % bcolors.BLUE
              + str(adversary.distance) + bcolors.ENDC)
        if args.metric in ["brightness", "rotation", "horizontal_translation",
                           "vertical_translation"]:
            print('Verifiable bound: %s' % bcolors.BLUE
                  + str(adversary.verifiable_bounds) + bcolors.ENDC)
        print('\n')

        if not os.path.exists('examples/images'):
            os.makedirs('examples/images')
        plot_image(adversary,
                   title=', '.join(keywords),
                   figname='examples/images/%s.png' % '_'.join(keywords))
    # paddlehub and pytorch object detection models
    elif args.model in summary['paddlehub_models'] or \
            args.model in summary['pytorchhub_models'] or \
            args.model in summary['keras_models']:
        print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
        print('Configuration:' + bcolors.CYAN + ' --framework %s '
                                                '--model %s --criterion %s '
                                                '--metric %s --distance %s' % tuple(keywords) + bcolors.ENDC)

        if adversary.image is not None:
            print('Minimum perturbation required: %s' % bcolors.BLUE
                  + str(adversary.distance) + bcolors.ENDC)
            print('\n')
            if not os.path.exists('examples/images'):
                os.makedirs('examples/images')
            if args.model in summary['keras_models']:
                class_names = model.get_class()
            else:
                class_names = None
            plot_image_objectdetection_hub(adversary, predictions, args.model, class_names=class_names, title=", ".join(keywords),
                                                 figname='examples/images/%s.png' % '_'.join(keywords))
        else:
            print(bcolors.BLUE + 'Attack failed' + bcolors.ENDC)
    # other object detection models
    else:
        print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
        print('Configuration:' + bcolors.CYAN + ' --framework %s '
                                                '--model %s --criterion %s '
                                                '--metric %s --distance %s' % tuple(keywords) + bcolors.ENDC)

        print('Minimum perturbation required: %s' % bcolors.BLUE
              + str(adversary.distance) + bcolors.ENDC)
        print('\n')

        plot_image_objectdetection(adversary, model, title=", ".join(keywords),
                                   figname='examples/images/%s.png' % '_'.join(keywords))


if __name__ == "__main__":
    summary = parse_summary()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help="specify the name of the model you want to evaluate",
        default="paddlehub_mobilenet_v2_animals"
    )
    parser.add_argument(
        "-e", "--metric",
        help="specify the name of the metric",
        default="salt_and_pepper_noise"
    )
    parser.add_argument(
        "-d", "--distance",
        help="specify the distance metric to evaluate adversarial examples",
        choices=summary['distances'],
        default="l2"
    )
    parser.add_argument(
        "-f", "--framework",
        help="specify the deep learning framework used by the target model",
        choices=summary['frameworks'],
        default="paddlehub"
    )
    parser.add_argument(
        "-c", "--criteria",
        help="specify the adversarial criteria for the evaluation",
        choices=summary['criterions'],
        default="misclassification"
    )
    parser.add_argument(
        "--verify",
        help="specify if you want a verifiable robustness bound",
        action="store_true"
    )
    parser.add_argument(
        "-t", "--max_iterations",
        help="specify the maximum number of iterations for the attack",
        type=int,
        default="1000"
    )
    parser.add_argument(
        "-i", "--image",
        help="specify the original input image for evaluation"
    )
    parser.add_argument(
        "--target_class",
        help="Used for detection/target-attack tasks, indicating the class you want to vanish",
        type=int
        # default=-1
    )
    args = parser.parse_args()
    validate_cmd(args)
    run_attack(args, summary)

    # #  given some images
    # folder_name = 'examples/'
    #
    # image_file_path = 'perceptron/utils/images/' + folder_name
    # for item in os.listdir(image_file_path):
    #
    #     print(item)
    #     args.image = folder_name + item
    #     validate_cmd(args)
    #     run_attack(args, summary)
