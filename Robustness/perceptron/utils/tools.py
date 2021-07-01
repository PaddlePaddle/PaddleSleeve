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

"""Launcher tools."""

from perceptron.utils.image import load_image
import matplotlib.pyplot as plt


class bcolors:
    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    CYAN = "\033[1;36m"
    GREEN = "\033[0;32m"
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


def get_image_format(framework_name, model_name):
    """Return the correct input range and shape for target framework and model"""
    special_shape = {'pytorch': {'inception_v3': (299, 299)},
                     'keras': {'xception': (299, 299),
                               'inception_v3': (299, 299),
                               'yolo_v3': (416, 416),
                               'ssd300': (300, 300)}}
    special_bound = {'keras': {'vgg16': (0, 255),
                               'vgg19': (0, 255),
                               'resnet50': (0, 255),
                               'ssd300': (0, 255)},
                     'cloud': {'aip_antiporn': (0, 255),
                               'google_safesearch': (0, 255),
                               'google_objectdetection': (0, 255)}}
    default_shape = (224, 224)
    default_bound = (0, 1)
    if special_shape.get(framework_name, None):
        if special_shape[framework_name].get(model_name, None):
            default_shape = special_shape[framework_name][model_name]
    if special_bound.get(framework_name, None):
        if special_bound[framework_name].get(model_name, None):
            default_bound = special_bound[framework_name][model_name]
    return {'shape': default_shape, 'bounds': default_bound}


def get_image(fpath, framework_name, model_name, data_format):
    """Get the image suitable for target model."""
    kwargs = get_image_format(framework_name, model_name)
    kwargs['data_format'] = data_format
    kwargs['path'] = fpath
    image = load_image(**kwargs)
    return image


def get_model(model_name, framework, summary):
    """Get model dispatcher."""
    switcher = {
        'keras': lambda: _load_keras_model(model_name, summary),
        'tensorflow': lambda: _load_keras_model(model_name, summary),
        'pytorch': lambda: _load_pytorch_model(model_name, summary),
        'cloud': lambda: _load_cloud_model(model_name, summary),
        'paddle': lambda: _load_paddle_model(model_name, summary)
    }
    _get_model = switcher.get(framework, None)
    return _get_model()


def get_distance(distance_name):
    """Get the distance metric."""
    import perceptron.utils.distances as distances
    switcher = {
        'mse': distances.MSE,
        'mae': distances.MAE,
        'linf': distances.Linf,
        "l0": distances.L0,
        "l2": distances.MSE
    }
    return switcher.get(distance_name, None)


def get_metric(attack_name, model, criteria, distance):
    """Get the attack class object."""
    import perceptron.benchmarks as metrics
    kwargs = {
        'model': model,
        'criterion': criteria,
        'distance': distance,
    }
    switcher = {
        "carlini_wagner_l2": lambda x: metrics.CarliniWagnerL2Metric(**x),
        "carlini_wagner_linf": lambda x: metrics.CarliniWagnerLinfMetric(**x),
        "additive_gaussian_noise": lambda x: metrics.AdditiveGaussianNoiseMetric(**x),
        "additive_uniform_noise": lambda x: metrics.AdditiveUniformNoiseMetric(**x),
        "blend_uniform_noise": lambda x: metrics.BlendedUniformNoiseMetric(**x),
        "gaussian_blur": lambda x: metrics.GaussianBlurMetric(**x),
        "brightness": lambda x: metrics.BrightnessMetric(**x),
        "contrast_reduction": lambda x: metrics.ContrastReductionMetric(**x),
        "motion_blur": lambda x: metrics.MotionBlurMetric(**x),
        "rotation": lambda x: metrics.RotationMetric(**x),
        "salt_and_pepper_noise": lambda x: metrics.SaltAndPepperNoiseMetric(**x),
        "spatial": lambda x: metrics.SpatialMetric(**x),
        "contrast": lambda x: metrics.ContrastReductionMetric(**x),
        "horizontal_translation": lambda x: metrics.HorizontalTranslationMetric(**x),
        "vertical_translation": lambda x: metrics.VerticalTranslationMetric(**x),
        "snow": lambda x: metrics.SnowMetric(**x),
        "fog": lambda x: metrics.FogMetric(**x),
        "frost": lambda x: metrics.FrostMetric(**x)
    }
    _init_attack = switcher.get(attack_name, None)
    attack = _init_attack(kwargs)
    return attack


def get_criteria(criteria_name, target_class=None):
    """Get the adversarial criteria."""
    import perceptron.utils.criteria as criteria
    switcher = {
        "misclassification": lambda: criteria.Misclassification(),
        "confident_misclassification": lambda: criteria.ConfidentMisclassification(),
        "topk_misclassification": lambda: criteria.TopKMisclassification(10),
        "target_class": lambda: criteria.TargetClass(target_class),
        "original_class_probability": lambda: criteria.OriginalClassProbability(),
        "target_class_probability": lambda: criteria.TargetClassProbability(target_class),
        "target_class_miss_google": lambda: criteria.TargetClassMissGoogle(target_class),
        "weighted_ap": lambda: criteria.WeightedAP(),
        "misclassification_antiporn": lambda: criteria.MisclassificationAntiPorn(),
        "misclassification_safesearch": lambda: criteria.MisclassificationSafeSearch(),
        "target_class_miss": lambda: criteria.TargetClassMiss(target_class),
    }
    return switcher.get(criteria_name, None)()


def _load_keras_model(model_name, summary):
    import keras.applications as models
    switcher = {
        'xception': lambda: models.xception.Xception(weights='imagenet'),
        'vgg16': lambda: models.vgg16.VGG16(weights='imagenet'),
        'vgg19': lambda: models.vgg19.VGG19(weights='imagenet'),
        "resnet50": lambda: models.resnet50.ResNet50(weights='imagenet'),
        "inception_v3": lambda: models.inception_v3.InceptionV3(weights='imagenet'),
        "yolo_v3": lambda: _load_yolov3_model(),
        "ssd300": lambda: _load_ssd300_model(),
        "retina_resnet_50": lambda: _load_retinanet_resnet50_model()
    }

    _load_model = switcher.get(model_name, None)
    _model = _load_model()

    from perceptron.models.classification.keras import KerasModel as ClsKerasModel
    from perceptron.models.detection.keras_ssd300 import KerasSSD300Model
    from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
    from perceptron.models.detection.keras_retina_resnet50 import KerasResNet50RetinaNetModel
    import numpy as np
    format = get_image_format('keras', model_name)
    if format['bounds'][1] == 1:
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        preprocessing = (mean, std)
    else:
        preprocessing = (np.array([104, 116, 123]), 1)
    switcher = {
        'yolo_v3': lambda x: KerasYOLOv3Model(x, bounds=(0, 1)),
        'ssd300': lambda x: KerasSSD300Model(x, bounds=(0, 255)),
        'retina_resnet_50': lambda x: KerasResNet50RetinaNetModel(None, bounds=(0, 255)),
    }
    _wrap_model = switcher.get(
        model_name,
        lambda x: ClsKerasModel(x, bounds=format['bounds'], preprocessing=preprocessing))
    kmodel = _wrap_model(_model)
    return kmodel


def _load_cloud_model(model_name, summary):
    import perceptron.models.classification.cloud as models
    switcher = {
        'aip_antiporn': lambda: _load_antiporn_model(),
        "google_safesearch": lambda: models.GoogleSafeSearchModel(),
        "google_objectdetection": lambda: models.GoogleObjectDetectionModel(),
    }

    _load_model = switcher.get(model_name, None)
    cmodel = _load_model()
    return cmodel


def _load_pytorch_model(model_name, summary):
    import torchvision.models as models
    switcher = {
        'alexnet': lambda: models.alexnet(pretrained=True).eval(),
        "vgg11": lambda: models.vgg11(pretrained=True).eval(),
        "vgg11_bn": lambda: models.vgg11_bn(pretrained=True).eval(),
        "vgg13": lambda: models.vgg13(pretrained=True).eval(),
        "vgg13_bn": lambda: models.vgg13_bn(pretrained=True).eval(),
        "vgg16": lambda: models.vgg16(pretrained=True).eval(),
        "vgg16_bn": lambda: models.vgg16_bn(pretrained=True).eval(),
        "vgg19": lambda: models.vgg19(pretrained=True).eval(),
        "vgg19_bn": lambda: models.vgg19_bn(pretrained=True).eval(),
        "resnet18": lambda: models.resnet18(pretrained=True).eval(),
        "resnet34": lambda: models.resnet34(pretrained=True).eval(),
        "resnet50": lambda: models.resnet50(pretrained=True).eval(),
        "resnet101": lambda: models.resnet101(pretrained=True).eval(),
        "resnet152": lambda: models.resnet152(pretrained=True).eval(),
        "squeezenet1_0": lambda: models.squeezenet1_0(pretrained=True).eval(),
        "squeezenet1_1": lambda: models.squeezenet1_1(pretrained=True).eval(),
        "densenet121": lambda: models.densenet121(pretrained=True).eval(),
        "densenet161": lambda: models.densenet161(pretrained=True).eval(),
        "densenet201": lambda: models.densenet201(pretrained=True).eval(),
        "inception_v3": lambda: models.inception_v3(pretrained=True).eval(),
    }

    _load_model = switcher.get(model_name, None)
    _model = _load_model()
    import torch
    if torch.cuda.is_available():
        _model = _model.cuda()
    from perceptron.models.classification.pytorch import PyTorchModel as ClsPyTorchModel
    import numpy as np
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    pmodel = ClsPyTorchModel(
        _model, bounds=(
            0, 1), num_classes=1000, preprocessing=(
            mean, std))
    return pmodel


def _load_paddle_model(model_name, summary):
    import paddle.vision.models as vm
    switcher = {
        "resnet18": lambda: vm.resnet18(pretrained=True),
        "resnet50": lambda: vm.resnet50(pretrained=True),
        "vgg16": lambda: vm.vgg16(pretrained=True)
    }
    _load_model = switcher.get(model_name, None)
    _model = _load_model()
    _model.eval()
    from perceptron.models.classification.paddle import PaddleModel as ClsPaddleModel
    import numpy as np
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    pmodel = ClsPaddleModel(_model, bounds=(
        0, 1), num_classes=1000, preprocessing=(
        mean, std))
    return pmodel


def _load_yolov3_model():
    from perceptron.zoo.yolov3.model import YOLOv3
    model = YOLOv3()
    return model


def _load_ssd300_model():
    from perceptron.zoo.ssd_300.keras_ssd300 import SSD300
    model = SSD300()
    return model


def _load_retinanet_resnet50_model():
    from perceptron.models.detection.keras_retina_resnet50 import KerasResNet50RetinaNetModel
    model = KerasResNet50RetinaNetModel()
    return model


def _load_antiporn_model():
    from perceptron.models.classification.cloud import AipAntiPornModel
    appId = '15064794'
    apiKey = '3R9pevnY2s077mCrzXP1Ole5'
    secretKey = "bl85bOsh49Ufp8VaMtYN3OX1pgKfehVp"
    credential = (appId, apiKey, secretKey)
    model = AipAntiPornModel(credential)
    return model


def plot_image(adversary, title=None, figname='compare.png'):
    """Plot the images."""
    prev = adversary.original_image
    after = adversary.image

    import numpy as np
    if prev.shape[0] == 3:
        prev = np.transpose(prev, (1, 2, 0))
        after = np.transpose(after, (1, 2, 0))

    max_value = 255 if prev.max() > 1 else 1

    diff = np.absolute(prev - after)
    scale = max_value / diff.max()
    diff = diff * scale

    if max_value == 255:
        prev = prev.astype('uint8')
        after = after.astype('uint8')
        diff = diff.astype('uint8')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.axis('off')

    ax1.imshow(prev)
    ax1.set_title('Origin')
    ax1.axis('off')

    ax2.imshow(after)
    ax2.set_title('Adversary')
    ax2.axis('off')

    ax3.imshow(diff)
    ax3.set_title('Diff * %.1f' % scale)
    ax3.axis('off')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.80)

    # in case you do not have GUI interface
    plt.savefig(figname, bbox_inches='tight')

    plt.show()


def plot_image_objectdetection(adversary, kmodel, bounds=(0, 1), title=None, figname='compare.png'):
    """Plot the images."""
    from perceptron.utils.image import draw_letterbox
    pred_ori = kmodel.predictions(adversary.original_image)
    pred_adv = kmodel.predictions(adversary.image)
    class_names = kmodel.get_class()

    ori_image = draw_letterbox(adversary.original_image, pred_ori, class_names=class_names, bounds=bounds)

    adv_image = draw_letterbox(adversary.image, pred_adv, class_names=class_names, bounds=bounds)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.axis('off')

    ax1.imshow(ori_image)
    ax1.set_title('Origin')
    ax1.axis('off')

    ax2.imshow(adv_image)
    ax2.set_title('Adversary')
    ax2.axis('off')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.9)

    # in case you do not have GUI interface
    plt.savefig(figname, bbox_inches='tight', dpi=1000)

    plt.show()


def load_pytorch_model(model_name):
    """Load pytorch model."""
    import torchvision.models as models
    switcher = {
        'alexnet': lambda: models.alexnet(pretrained=True).eval(),
        "vgg11": lambda: models.vgg11(pretrained=True).eval(),
        "vgg11_bn": lambda: models.vgg11_bn(pretrained=True).eval(),
        "vgg13": lambda: models.vgg13(pretrained=True).eval(),
        "vgg13_bn": lambda: models.vgg13_bn(pretrained=True).eval(),
        "vgg16": lambda: models.vgg16(pretrained=True).eval(),
        "vgg16_bn": lambda: models.vgg16_bn(pretrained=True).eval(),
        "vgg19": lambda: models.vgg19(pretrained=True).eval(),
        "vgg19_bn": lambda: models.vgg19_bn(pretrained=True).eval(),
        "resnet18": lambda: models.resnet18(pretrained=True).eval(),
        "resnet34": lambda: models.resnet34(pretrained=True).eval(),
        "resnet50": lambda: models.resnet50(pretrained=True).eval(),
        "resnet101": lambda: models.resnet101(pretrained=True).eval(),
        "resnet152": lambda: models.resnet152(pretrained=True).eval(),
        "squeezenet1_0": lambda: models.squeezenet1_0(pretrained=True).eval(),
        "squeezenet1_1": lambda: models.squeezenet1_1(pretrained=True).eval(),
        "densenet121": lambda: models.densenet121(pretrained=True).eval(),
        "densenet161": lambda: models.densenet161(pretrained=True).eval(),
        "densenet201": lambda: models.densenet201(pretrained=True).eval(),
        "inception_v3": lambda: models.inception_v3(pretrained=True).eval(),
    }

    _load_model = switcher.get(model_name, None)
    _model = _load_model()
    return _model
