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
HopSkipJumpAttack tutorial on ImageNet.
"""
from __future__ import print_function
import os
import sys
import cv2
sys.path.append("../..")

import logging
logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import argparse
import numpy as np
import functools
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize
from paddle.vision.transforms import ToTensor
from past.utils import old_div
from adversary import Adversary
from examples.utils import add_arguments, print_arguments, show_images_diff
from examples.utils import bcolors
from attacks.hop_skip_jump_attack import HopSkipJumpAttack
from models.blackbox import PaddleBlackBoxModel
from models.whitebox import PaddleWhiteBoxModel
from paddle.vision.models import resnet18, resnet50
import tensorflow as tf
from tf2onnx import convert
from ResNet import ResNet
import config as c
from utils.data_utils import load_image
import torch
import torchvision.models as models
from paddle.vision import transforms as pt
from PIL import Image
import onnxruntime
from torchvision import transforms

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('target_image', str, None, 'The type of successful attack, e.g., input/schoolbus.png')
add_arg('image_path', str, 'input/pickup_truck.jpeg', 'given the image path, e.g., input/schoolbus.png')
add_arg('num_iterations', int, 1, 'iter num for hsja')
add_arg('norm', str, 'l2', 'choose between [l2, linf]')

'''
add_arg('origin_model_path', str, "/work/model_to_onnx/paddle/ResNet50_infer/resnet50.pdparams", "origin model path")
add_arg('onnx_model_path', str, "/work/model_to_onnx/models/paddle_res50", "onnx model path")
add_arg('framework', str, "paddle", "Deep learning framework, paddle/tensorflow/pytorch")
'''

'''
add_arg('origin_model_path', str, "/work/model_to_onnx/tensorflow/ResNet_18.h5", "origin model path")
add_arg('onnx_model_path', str, "/work/model_to_onnx/models/tf_resnet18.onnx", "onnx model path")
add_arg('framework', str, "tensorflow", "Deep learning framework, paddle/tensorflow/pytorch")
'''

 
add_arg('origin_model_path', str, "/work/model_to_onnx/torch/resnet18.pth", "origin model path")
add_arg('onnx_model_path', str, "/work/model_to_onnx/models/torch_res18.onnx", "onnx model path")
add_arg('framework', str, "pytorch", "Deep learning framework, paddle/tensorflow/pytorch")


args = parser.parse_args()
print_arguments(args)

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

def onnx_predict_paddle(img_path, onnx_model_path):
    preprocess = pt.Compose([
    pt.Resize(256),
    pt.CenterCrop(224),
    pt.ToTensor(),
    pt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = paddle.unsqueeze(img_t, 0)
    batch_t = paddle.to_tensor(batch_t)
    ort_sess = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_sess.get_inputs()[0].name: batch_t.cpu().numpy()}
    ort_outs = ort_sess.run(None, ort_inputs)
    ort_outs = softmax(ort_outs[0][0])
    predict =  np.argmax(ort_outs, 0)
    predict_label = predict.item()
    return predict_label


def onnx_predict_tf(img_path, onnx_model_path):
    def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    img, _ = load_image(tf.constant(img_path), 0)
    image = np.expand_dims(img, axis=0)
    providers = ['CPUExecutionProvider']
    ort_sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    img =  {ort_sess.get_inputs()[0].name: image}
    # onnx�~N��~P~F
    output_names = ["output"]
    onnx_pred = ort_sess.run(None, img)
    onnx_label = np.argmax(onnx_pred[0][0])
    return onnx_label

def onnx_predict_torch(img_path, onnx_model_path):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to('cpu')
    ort_sess = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_sess.get_inputs()[0].name: batch_t.cpu().numpy()}
    ort_outs = ort_sess.run(None, ort_inputs)
    ort_outs = softmax(ort_outs[0][0])
    predict = np.argmax(ort_outs, 0)
    predict_label = predict.item()
    return predict_label

def read_target_image():
    """

    Returns:

    """
    img_ori = cv2.imread(args.target_image)
    im = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224))
    im = (im.T / 255).astype(np.float32)

    orig = img_ori[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = old_div((img - mean), std)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img_tensor = paddle.to_tensor(img, dtype='float32', stop_gradient=False)
    return im, img_tensor

def main(orig):
    """

    Args:
        orig: input image, type: ndarray, size: h*w*c
        method: denoising method
    Returns:

    """

    # Define what device we are using
    logging.info("CUDA Available: {}".format(paddle.is_compiled_with_cuda()))
    img = orig.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = old_div((img - mean), std)

    img = img.transpose(2, 0, 1)
    C, H, W = img.shape
    img = np.expand_dims(img, axis=0)
    img = paddle.to_tensor(img, dtype='float32', stop_gradient=False)

    # Initialize the network
    model = paddle.vision.models.resnet101(pretrained=True, num_classes=1000)
    model.eval()
    # init a paddle model
    paddle_model = PaddleBlackBoxModel(
        [model],
        [1],
        (0, 1),
        mean=mean,
        std=std,
        input_channel_axis=0,
        input_shape=(3, 224, 224),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=1000)

    predict = model(img)[0]
    label = np.argmax(predict)
    img = np.squeeze(img)
    inputs = img
    labels = label

    # Read the labels file for translating the labelindex to english text
    with open('../../../Robustness/perceptron/utils/labels.txt') as info:
        imagenet_dict = eval(info.read())
    print(bcolors.CYAN + "input image label: {}".format(imagenet_dict[label]) + bcolors.ENDC)
    print(bcolors.CYAN + "input image label: {}".format(label) + bcolors.ENDC)
    adversary = Adversary(inputs.numpy(), labels)
    # non-targeted attack
    attack_config = {"steps": 100}
    print()
    if args.target_image:
        target_image, target_image_tensor = read_target_image()
        predict = model(target_image_tensor)[0]
        tlabel = np.argmax(predict)
        print(bcolors.CYAN + "target image label: {}".format(imagenet_dict[tlabel]) + bcolors.ENDC)
        print(bcolors.CYAN + "target image label: {}".format(tlabel) + bcolors.ENDC)        
        params = {
            'target_label': tlabel,
            'target_image': target_image,
            'constraint': args.norm,
            'num_iterations': args.num_iterations
        }        
        attack = HopSkipJumpAttack(paddle_model, params=params)
        adversary.set_status(is_targeted_attack=True, target_label=tlabel)
    else:
        params = {
            'target_label': None,
            'target_image': None,
            'constraint': args.norm,
            'num_iterations': args.num_iterations
        }        
        attack = HopSkipJumpAttack(paddle_model, params=params)

    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(bcolors.RED + "HSJA succeeded, adversarial_label: {}".format( \
            imagenet_dict[adversary.adversarial_label]) + bcolors.ENDC)
        print(bcolors.RED + "HSJA succeeded, adversarial_label: {}".format( \
            adversary.adversarial_label) + bcolors.ENDC)
        adv = adversary.adversarial_example
        adv = np.squeeze(adv)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        adv_cv = np.copy(adv)
        adv_cv = adv_cv[..., ::-1]  # RGB to BGR
        cv2.imwrite('output/img_adv_hsja.png', adv_cv)
        show_images_diff(orig, labels, adv, adversary.adversarial_label)
    else:
        print('attack failed')

    origin_model_path = args.origin_model_path
    onnx_model_path = args.onnx_model_path
    framework = args.framework
    if framework == 'paddle':
        # load origin model
        paddle.set_device('cpu')
        param_state_dict = paddle.load(origin_model_path)
        if '18' in origin_model_path:
            model = resnet18(pretrained=False, num_classes=1000)
        elif '50' in origin_model_path:
            model = resnet50(pretrained=False, num_classes=1000)
        model.set_dict(param_state_dict)
        model.eval()
        device = paddle.get_device()
        paddle.set_device(device)
        # origin to onnx
        # paddle.set_device('gpu:0' if paddle.is_compiled_with_cuda() else 'cpu')
        x_spec = paddle.static.InputSpec([1, 3, 224, 224], 'float32', 'x')
        paddle.onnx.export(model, onnx_model_path, input_spec=[x_spec], opset_version=11)
        print('paddle2onnx sucess')
        onnx_model_path = onnx_model_path + '.onnx'
        origin_onnx_label = onnx_predict_paddle(args.image_path, onnx_model_path)
        attack_onnx_label = onnx_predict_paddle('output/img_adv_hsja.png', onnx_model_path)
    elif framework == 'tensorflow':
        # load origin model
        if '18' in origin_model_path:
            model = ResNet(18)
        elif '50' in origin_model_path:
            model = ResNet(50)
        model.build(input_shape=(None,) + c.input_shape)
        model.load_weights(origin_model_path)
         # origin to onnx
        # logging.basicConfig(level=logging.INFO)
        convert.from_keras(model, input_signature=[tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)], output_path=onnx_model_path, opset=16)  # �~@~B�~E~Monnxruntime�~Z~Dopset�~I~H�~\�)
        print('tensorflow2onnx sucess')
        origin_onnx_label = onnx_predict_tf(args.image_path, onnx_model_path)
        attack_onnx_label = onnx_predict_tf('output/img_adv_hsja.png', onnx_model_path)
    elif framework == 'pytorch':
        # load origin model
        if '18' in origin_model_path:
            model = models.resnet18(weights=None)
        elif '50' in origin_model_path:
            model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(origin_model_path, weights_only=False))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        # origin to onnx
        example_input = torch.randn(1, 3, 224, 224).to(device)
        # verbose=False/True for loging
        torch.onnx.export(model, example_input, onnx_model_path, opset_version=11, verbose=False)
        print('pytorch2onnx sucess')
        origin_onnx_label = onnx_predict_torch(args.image_path, onnx_model_path)
        attack_onnx_label = onnx_predict_torch('output/img_adv_hsja.png', onnx_model_path)

    print('origin image onnx_model label is {}, attack image onnx_model label is {}'.format(origin_onnx_label, attack_onnx_label))


if __name__ == '__main__':
    # read image
    orig = cv2.imread(args.image_path)
    orig = orig[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    # denoise
    main(orig)


