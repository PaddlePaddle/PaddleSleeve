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
CW tutorial on imagenet using the attack tool.
"""
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../..")

from past.utils import old_div
import logging
logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)


import argparse
import cv2
import functools
import numpy as np
import paddle
from models.whitebox import PaddleWhiteBoxModel
from paddle.vision.models import resnet18, resnet50

from adversary import Adversary
from attacks.cw import CW_L2
from examples.utils import add_arguments, print_arguments, show_images_diff
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
add_arg('target', int, 126, "target class.")
add_arg('class_dim', int, 1000, "Class number.")
add_arg('image_shape', str, "3,224,224", "Input image size")


add_arg('origin_model_path', str, "/work/model_to_onnx/paddle/ResNet50_infer/resnet50.pdparams", "origin model path")
add_arg('onnx_model_path', str, "/work/model_to_onnx/models/paddle_res50", "onnx model path")
add_arg('framework', str, "paddle", "Deep learning framework, paddle/tensorflow/pytorch")


'''
add_arg('origin_model_path', str, "/work/model_to_onnx/tensorflow/ResNet_50.h5", "origin model path")
add_arg('onnx_model_path', str, "/work/model_to_onnx/models/tf_resnet50.onnx", "onnx model path")
add_arg('framework', str, "tensorflow", "Deep learning framework, paddle/tensorflow/pytorch")
'''

''' 
add_arg('origin_model_path', str, "/work/model_to_onnx/torch/resnet50.pth", "origin model path")
add_arg('onnx_model_path', str, "/work/model_to_onnx/models/torch_res50.onnx", "onnx model path")
add_arg('framework', str, "pytorch", "Deep learning framework, paddle/tensorflow/pytorch")
'''

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
    # 读取onnx模型，安装GPUonnx，并设置providers = ['GPUExecutionProvider']，可以实现GPU运行onnx
    providers = ['CPUExecutionProvider']
    ort_sess = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    img =  {ort_sess.get_inputs()[0].name: image}
    # onnx推理
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

def main(image_path):
    """

    Args:
        image_path: path of image to be test

    Returns:

    """
    # parse args
    args = parser.parse_args()
    print_arguments(args)
    image_shape = [int(m) for m in args.image_shape.split(",")]
    class_dim = args.class_dim

    # Define what device we are using
    logging.info("CUDA Available: {}".format(paddle.is_compiled_with_cuda()))

    orig = cv2.imread(image_path)[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = paddle.vision.transforms.Normalize(mean, std)

    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = paddle.to_tensor(img, dtype='float32', stop_gradient=False)
    img = norm(img)
    img = paddle.unsqueeze(img, axis=0)

    # Initialize the network
    model = paddle.vision.models.resnet50(pretrained=True)
    model.eval()

    predict = model(img)[0]
    print(predict.shape)
    label = np.argmax(predict)
    print("label={}".format(label))

    # init a paddle model
    paddle_model = PaddleWhiteBoxModel(
        [model],
        [1],
        (0, 1),
        mean=mean,
        std=std,
        input_channel_axis=0,
        input_shape=(3, 224, 224),
        loss=paddle.nn.CrossEntropyLoss(),
        nb_classes=1000)

    img = np.squeeze(img)
    inputs = img
    labels = label #orig_label

    print("input img shape: ", inputs.shape)

    adversary = Adversary(inputs.numpy(), labels)

    # targeted attack
    target_class = args.target
    if target_class != -1:
        tlabel = target_class
        adversary.set_status(is_targeted_attack=True, target_label=tlabel)

    attack = CW_L2(paddle_model)

    attack_config = {"attack_iterations": 100,
                     "c_search_steps": 20}

    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
              % adversary.adversarial_label)

        adv = adversary.adversarial_example
        adv = np.squeeze(adv)
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        adv_cv = np.copy(adv)
        adv_cv = adv_cv[..., ::-1]  # RGB to BGR
        cv2.imwrite('output/img_adv_cw.png', adv_cv)

        show_images_diff(orig, labels, adv, adversary.adversarial_label)
    else:
        print('attack failed')

    print("cw attack done")
    
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
        origin_onnx_label = onnx_predict_paddle(image_path, onnx_model_path)
        attack_onnx_label = onnx_predict_paddle('output/img_adv_cw.png', onnx_model_path)
    elif framework == 'tensorflow':
        # load origin model
        if '18' in origin_model_path:
            model = ResNet(18)
        elif '50' in origin_model_path:
            model = ResNet(50)
        model.build(input_shape=(None,) + c.input_shape)
        model.load_weights(origin_model_path)
         # origin to onnx
        # 控制是否打印tf2onnx日志
        # logging.basicConfig(level=logging.INFO)
        # 转换为ONNX格式，指定输入形状（1,224,224,3）
        convert.from_keras(model, input_signature=[tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)], output_path=onnx_model_path, opset=16)  # 适配onnxruntime的opset版本)
        print('tensorflow2onnx sucess')
        origin_onnx_label = onnx_predict_tf(image_path, onnx_model_path)
        attack_onnx_label = onnx_predict_tf('output/img_adv_cw.png', onnx_model_path)
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
        #控制打印torch.onnx日志，verbose=False不打印，verbose=True打印
        torch.onnx.export(model, example_input, onnx_model_path, opset_version=11, verbose=False)
        print('pytorch2onnx sucess')
        origin_onnx_label = onnx_predict_torch(image_path, onnx_model_path)
        attack_onnx_label = onnx_predict_torch('output/img_adv_cw.png', onnx_model_path)

    print('origin image onnx_model label is {}, attack image onnx_model label is {}'.format(origin_onnx_label, attack_onnx_label))

if __name__ == '__main__':
    #main("input/schoolbus.png")
    #main("input/cropped_panda.jpeg")
    main("input/pickup_truck.jpeg")
