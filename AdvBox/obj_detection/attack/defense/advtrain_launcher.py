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
paddle2 model adversarial training
"""
import argparse
import os
import sys

sys.path.append('../..')
from attack.utils.tools import bcolors
import paddle
from ppdet.core.workspace import create, load_config
from ppdet.metrics import Metric, COCOMetric
import attack.single_attack as attacks
from attack.models import paddledet as models
from attack import defense


def _init_metric(cfg, dataset):
    classwise = cfg['classwise'] if 'classwise' in cfg else False
    bias = cfg['bias'] if 'bias' in cfg else 0
    output_eval = cfg['output_eval'] \
        if 'output_eval' in cfg else None
    save_prediction_only = cfg.get('save_prediction_only', False)

    # pass clsid2catid info to metric instance to avoid multiple loading
    # annotation file
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}

    # when do validation in train, annotation file should be get from
    # EvalReader instead of dataset(which is TrainReader)
    anno_file = dataset.get_anno()
    dataset = dataset

    IouType = cfg['IouType'] if 'IouType' in cfg else 'bbox'
    _metrics = [
        COCOMetric(
            anno_file=anno_file,
            clsid2catid=clsid2catid,
            classwise=classwise,
            output_eval=output_eval,
            bias=bias,
            IouType=IouType,
            save_prediction_only=save_prediction_only)
    ]

    return _metrics


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


def main(args, model):
    from attack.defense.reader import TrainReader
    """
    Main function for running adversarial training.
    Returns:
        None
    """
    attack_config1 = {"p": 0.5,
                      "use_opt": True}
    init_config = {"distance": args.distance,
                   'verbose': False}
    attack_config2 = {"p": 0.5,
                      "max_iterations": 15,
                      "c_search_steps": 6,
                      "abort_early": True}

    model_path = os.path.dirname(os.path.realpath(__file__ + '/..')) + '/outputs/models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    adversarial_trans = defense.DetectionAdversarialTransform(model,
                                                              [attacks.ProjectedGradientDescentMetric,
                                                               attacks.CarliniWagnerMetric],
                                                              [init_config, init_config],
                                                              [attack_config1, attack_config2])

    cfg = model._cfg
    train_dataset = cfg['TrainDataset']

    # Denoiser need special reader
    if args.defense == 'denoiser':
        cfg['TrainReader']['batch_size'] = 1
        cfg['TrainReader'].pop('mixup_epoch')
        cfg['TrainReader']['sample_transforms'] = cfg['TrainReader']['sample_transforms'][0:1]

    loader_kwargs = {}
    loader_kwargs.update(cfg['TrainReader'])
    train_loader = TrainReader(**loader_kwargs)(train_dataset, cfg.worker_num)

    # build optimizer in train mode
    steps_per_epoch = len(train_loader)
    lr = create('LearningRate')(steps_per_epoch)
    optimizer = create('OptimizerBuilder')(lr, model._model)
    collate_fn = paddle.fluid.dataloader.collate.default_collate_fn
    denoiser = defense.denoiser.DUNET()

    # adjust named parameters according to training method
    if args.defense == 'natural_advtrain':
        adversarial_train = defense.adversarial_train_natural
        train_configs = {"epoch_num": 2,
                         "advtrain_start_num": 0,
                         "adversarial_trans": adversarial_trans,
                         'optimizer': optimizer,
                         "scheduler": lr,
                         "collate_batch": collate_fn}
    elif args.defense == 'free_advtrain':
        adversarial_train = defense.free_advtrain
        train_configs = {"epoch_num": 50,
                         'steps': 5,
                         "eps": 1 / 255,
                         "scheduler": lr,
                         "collate_batch": collate_fn}
    else:
        print('Guided denoiser training will be supported in the future')
        exit(-1)

        adversarial_train = defense.hgd_training
        train_configs = {"epoch_num": 50,
                         "adversarial_trans": adversarial_trans,
                         "eps": 1 / 255,
                         "denoiser": denoiser,
                         # "resume_epoch": 19,
                         "loss": 'fgd',
                         "collate_batch": collate_fn}

    val_acc_history, val_loss_history, model, opt = adversarial_train(model=model._model,
                                                                      train_loader=train_loader,
                                                                      test_loader=None,
                                                                      save_path=model_path,
                                                                      **train_configs)


if __name__ == '__main__':
    summary = parse_summary()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help="specify the config file of the model you want to defend",
        default="yolov3_darknet53_advtrain"
    )
    parser.add_argument(
        "-e", "--defense",
        help="specify the defense method",
        choices=['natural_advtrain',
                 'free_advtrain',
                 'denoiser'],
        default="free_advtrain"
    )
    parser.add_argument(
        "-d", "--distance",
        help="specify the distance metric to evaluate adversarial examples",
        choices=summary['distances'],
        default="linf"
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
        "--pretrained",
        help="specify whether use pretrained weight",
        default=False
    )
    parser.add_argument(
        "--pretrained_weight",
        help="specify the pretrained weight of the model, only valid when pretrained is set to False",
        default='ppdet://models/yolov3_darknet53_270e_coco.pdparams'
    )
    args = parser.parse_args()

    config_dir = os.path.dirname(os.path.realpath(__file__ + '/../..')) + '/configs/'
    model = None
    if 'yolov3' in args.model:
        config_file = config_dir + 'yolov3/{}.yml'.format(args.model)
        if not os.path.isfile(config_file):
            print(bcolors.RED + 'Model config file does not exist. Please provide a valid model: ' + bcolors.ENDC)
            exit(-1)
        model = models.PPdet_Yolov3_Model(args.model, pretrained=args.pretrained)
        if not args.pretrained and args.pretrained_weight is not None:
            model.load_weight(args.pretrained_weight)
    elif 'faster_rcnn' in args.model:
        config_file = config_dir + 'faster_rcnn/{}.yml'.format(args.model)
        if not os.path.isfile(config_file):
            print(bcolors.RED + 'Model config file does not exist. Please provide a valid model: ' + bcolors.ENDC)
            exit(-1)
        model = models.PPdet_Rcnn_Model(args.model, cascade=False, pretrained=args.pretrained)
        if not args.pretrained and args.pretrained_weight is not None:
            model.load_weight(args.pretrained_weight)
    elif 'cascade_rcnn' in args.model:
        config_file = config_dir + 'cascade_rcnn/{}.yml'.format(args.model)
        if not os.path.isfile(config_file):
            print(bcolors.RED + 'Model config file does not exist. Please provide a valid model: ' + bcolors.ENDC)
            exit(-1)
        model = models.PPdet_Rcnn_Model(args.model, cascade=True, pretrained=args.pretrained)
        if not args.pretrained and args.pretrained_weight is not None:
            model.load_weight(args.pretrained_weight)
    elif 'detr' in args.model:
        config_file = config_dir + 'detr/{}.yml'.format(args.model)
        if not os.path.isfile(config_file):
            print(bcolors.RED + 'Model config file does not exist. Please provide a valid model: ' + bcolors.ENDC)
            exit(-1)
        model = models.PPdet_Detr_Model(args.model, pretrained=args.pretrained)
        if not args.pretrained and args.pretrained_weight is not None:
            model.load_weight(args.pretrained_weight)
    else:
        print(bcolors.RED + 'Please provide a valid model: ' + bcolors.ENDC)
        exit(-1)

    if model is not None:
        main(args, model)

