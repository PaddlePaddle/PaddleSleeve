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

sys.path.append("../..")
from attack.utils.tools import get_metric, get_distance, get_criteria, get_model
from attack.utils.tools import plot_image_objectdetection_ppdet, bcolors
import argparse
import numpy as np

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def validate_cmd(args, summary):
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
    if args.framework == 'paddledet':
        valid_detection_models = summary['detection_models']
    
    is_objectdetection = (args.framework == 'paddledet' and args.model in valid_detection_models)

    if is_objectdetection:
        hint_criterion_string = 'Your should choose criterion that supports object detection model.'
        valid_criterions = summary['detection_criterions']
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
        print('For example: --target_class 0')
        exit(-1)

    # step 5. check the model whether provide gradient when using cw attack
    must_white_box = args.metric in summary['whitebox_metrics']
    if must_white_box and args.framework != 'paddledet':
        valid_models = summary['paddledet_models']
        print(bcolors.WARNING + 'Your metric requires a white-box model.' + bcolors.ENDC)
        print(bcolors.RED + 'Please provide a valid model: ' + bcolors.ENDC +
            ', '.join(valid_models))
        exit(-1)


def parse_summary():
    """Parse dictionary from summary json file."""
    import json
    import os
    with open(
            os.path.dirname(os.path.realpath(__file__ + '/..')) +
            '/utils/summary.json') as f:
        summary = json.load(f)

    summary['models'] = list(set(summary['paddledet_models']))
    return summary


def run_attack(args, summary):
    """Run the attack."""
    model = get_model(args.model, args.framework, summary)
    distance = get_distance(args.distance)

    channel_axis = 'channels_first' if args.framework == 'paddledet' else 'channels_last'
    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__ + '/..')), 'utils/images/%s' % args.image)
    if args.framework == 'paddledet':
        image = model.load_image([image_path]).numpy()
    else:
        # May add other dataloaders here
        pass

    if args.framework == 'paddledet':
        original_pred = model.predictions(image)
    else:
        # May add more frameworks here
        pass

    if args.framework == 'paddledet':
        if args.target_class != -1:
            target_class = args.target_class
        else:
            target_class = int(original_pred[np.argmax(original_pred[:, 1]), 0])
        criteria = get_criteria(args.criteria, target_class, model_name=args.model)
    else:
        pass

    metric = get_metric(args.metric, model, criteria, distance)

    # May manually add attacking configs for specific attack method if desired
    kwargs = {'abort_early': True}

    print(bcolors.BOLD + 'Process start' + bcolors.ENDC)
    adversary = metric(image, original_pred, unpack=False, **kwargs)

    if adversary.image is None:
        print(bcolors.WARNING + 'Warning: Cannot find an adversary!' + bcolors.ENDC)
        return adversary

    ###################  print summary info  #####################################
    keywords = [args.framework, args.model, args.criteria, args.metric, args.distance]

    # Result visualization for paddledet models
    if args.framework == 'paddledet':
        print(bcolors.HEADER + bcolors.UNDERLINE + 'Summary:' + bcolors.ENDC)
        print('Configuration:' + bcolors.CYAN + ' --framework %s '
                                                '--model %s --criterion %s '
                                                '--metric %s --distance %s' % tuple(keywords) + bcolors.ENDC)

        print('Minimum perturbation required: %s' % bcolors.BLUE
              + str(adversary.distance) + bcolors.ENDC)
        print('\n')

        img_dir = os.path.dirname(os.path.realpath(__file__ + '/..')) + '/outputs/images'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        plot_image_objectdetection_ppdet(adversary, model,
                                         title=", ".join(keywords),
                                         figname='{0}/{1}.png'.format(img_dir, '_'.join(keywords)))
        print(bcolors.BLUE + 'Process Finished' + bcolors.ENDC)
        print('Visualization result is saved in %s' % img_dir)
        print('\n')
    else:
        pass

if __name__ == "__main__":
    summary = parse_summary()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help="specify the name of the model you want to evaluate",
        default="paddledet_yolov3_darknet53"
    )
    parser.add_argument(
        "-e", "--metric",
        help="specify the name of the metric",
        default="pgd"
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
        default="paddledet"
    )
    parser.add_argument(
        "-c", "--criteria",
        help="specify the adversarial criteria for the evaluation",
        choices=summary['criterions'],
        default="target_class_miss"
    )
    parser.add_argument(
        "-i", "--image",
        help="specify the original input image for evaluation"
    )
    parser.add_argument(
        "--target_class",
        help="Used for detection/target-attack tasks, indicating the class you want to vanish",
        type=int,
        default=-1
    )
    args = parser.parse_args()
    validate_cmd(args, summary)
    run_attack(args, summary)

