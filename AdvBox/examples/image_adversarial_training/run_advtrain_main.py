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
paddle2 model adversarial training demo on CIFAR10 data
"""
import os
import sys
sys.path.append("../..")
import argparse
import functools

import paddle
from defences.adversarial_transform import ClassificationAdversarialTransform
from models.whitebox import PaddleWhiteBoxModel
from examples.utils import add_arguments, print_arguments, get_best_weigthts_from_folder
from main_setting import model_zoo, training_zoo, dataset_zoo, attack_zoo, get_model_setting, get_save_path, get_train_method_setting, get_dataset, get_attack_setting, assert_input

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model', str, 'preactresnet', 'Model, choose in {model_zoo}.'.format(model_zoo=model_zoo))
add_arg('training_method', str, 'base', 'Training method, choose in {training_zoo}.'.format(training_zoo=training_zoo))
add_arg('attack_method', str, 'FGSM', 'Attack method, choose in {attack_zoo}. Only works if the "training_method" is not "base"'.format(attack_zoo=attack_zoo))
add_arg('dataset', str, 'cifar10', 'Dataset, choose in {dataset_zoo}.'.format(dataset_zoo=dataset_zoo))
add_arg('use_base_pretrain', str, 'no', 'Whether to use a model pre-trained in base mode, choose in ("yes", "no"), if training_method is "base" only support "no". If "yes", the best model trained in "base" mode will be used, so "base" mode must be used first.')

def main():
    """
    Main function for running adversarial training.
    Returns:
        None
    """
    args = parser.parse_args()
    print_arguments(args)

    model_choice = args.model
    training_choice = args.training_method
    dataset_choice = args.dataset
    attack_choice = args.attack_method
    use_base_pretrain = args.use_base_pretrain

    assert_input(model_choice, training_choice, dataset_choice, attack_choice, use_base_pretrain)

    save_path = get_save_path(model_choice, training_choice, dataset_choice, attack_choice, use_base_pretrain)

    model, MEAN, STD, train_transform, test_transform = get_model_setting(model_choice, dataset_choice)
    if use_base_pretrain == 'yes':
        assert training_choice != 'base', '"base" training mothed not support use_base_pretrain "yes".'
        pretrain_model_folder = "./tutorial_result/%s/%s/base" % (dataset_choice, model_choice)
        assert os.path.exists(pretrain_model_folder), 'Not exists %s. Using a pretrained model requires first training the pretrained model using the base method' % pretrain_model_folder
        pretrain_model_path = get_best_weigthts_from_folder(pretrain_model_folder, "base_net_")
        model_state_dict = paddle.load(pretrain_model_path)
        model.set_state_dict(model_state_dict)

    if training_choice == 'base':
        attack_method = None
        init_config = None
        attack_config = None
    else:
        attack_method, init_config, attack_config = get_attack_setting(attack_choice)
    
    adverarial_train, enhance_config, advtrain_settings = get_train_method_setting(model, training_choice)
    if attack_config is not None:
        for k, v in attack_config.items():
            enhance_config[k] = v
        

    train_set, num_class = get_dataset(dataset_choice, 'train', train_transform)
    test_set, _ = get_dataset(dataset_choice, 'test', test_transform)
    
    test_loader = paddle.io.DataLoader(test_set, batch_size=1)
    data = test_loader().next()

    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [model],
        [1],
        (0, 1),
        mean=MEAN,
        std=STD,
        input_channel_axis=0,
        input_shape=tuple(data[0].shape[1:]),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=num_class)

    adversarial_trans = ClassificationAdversarialTransform(paddle_model, [attack_method],
                                                           [init_config], [enhance_config])
    advtrain_settings["adversarial_trans"] = adversarial_trans
    adverarial_train(model, train_set, test_set,
        save_path=save_path, **advtrain_settings)


if __name__ == '__main__':
    main()
