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
The Target Ghosting Attack demonstration.
Contains:
* Initialize a yolo detector and inference pictures.
* Generate perturbation using model weights.
* Generate perturbed image.

Author: xiongjunfeng
"""
# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import os
import cv2
from PIL import Image

import paddle
from ppdet.core.workspace import create
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.data.source.category import get_categories
from ppdet.metrics import get_infer_results
from depreprocess.operator_composer import OperatorCompose
import paddle.nn.functional as F
import copy

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--target_img",
        type=str,
        default=None,
        help="Image path, infer image with masked on.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_txt",
        type=bool,
        default=False,
        help="Whether to save inference result in txt.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def get_pcls(model, neck_feats):
    """
    Get pcls with given neck_feats.
    Args:
        model: paddle.model. PaddleDetection model.
        neck_feats: paddle.tensor. Inferenced result from detector.head.

    Returns:
        paddle.tensor. pcls tensor.
    """
    # assert len(feats) == len(self.anchors)
    pcls_list = []

    for i, feat in enumerate(neck_feats):
        yolo_output = model.yolo_head.yolo_outputs[i](feat)
        if model.data_format == 'NHWC':
            yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])

        p = yolo_output
        number_anchor = 3
        b, c, h, w = p.shape

        p = p.reshape((b, number_anchor, -1, h, w)).transpose((0, 1, 3, 4, 2))
        # x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        # w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        pcls_list.append(pcls)

    return pcls_list


def pcls_kldivloss(pcls_list, target_pcls_list):
    """
    Compute the kl distance between pcls and target pcls.
    Args:
        pcls_list: list. Middle output from yolo loss. pcls is the classification feature map.
        target_pcls_list: list. The target pcls feature map.

    Returns:
        paddle.tensor. kl distance.
    """
    kldiv_criterion = paddle.nn.KLDivLoss(reduction='batchmean')
    logsoftmax = paddle.nn.LogSoftmax()
    softmax = paddle.nn.Softmax()
    kldivloss = 0

    for pcls, target_pcls in zip(pcls_list, target_pcls_list):
        loss_kl = kldiv_criterion(logsoftmax(pcls), softmax(target_pcls))
        kldivloss += loss_kl

    return kldivloss


# TODO: Method 2, try to use logits dispersion.
def yolo_featuremap_ld_attack(model, neck_feats, attack_step=10, epsilon_ball=100/255):
    """
    Unsupervised attack. Maximze kl distance from original feature map.
    Args:
        model:
        neck_feats:
        attack_step:
        epsilon_ball:

    Returns:

    """
    # assert len(feats) == len(self.anchors)
    yolo_outputs = []
    kldiv_criterion = paddle.nn.KLDivLoss(reduction='batchmean')
    logsoftmax = paddle.nn.LogSoftmax()
    softmax = paddle.nn.Softmax()

    for i, feat in enumerate(neck_feats):
        yolo_output = model.yolo_head.yolo_outputs[i](feat)
        if model.data_format == 'NHWC':
            yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
        yolo_outputs.append(yolo_output)
        p = yolo_output
        number_anchor = 3
        b, c, h, w = p.shape

    step_size = epsilon_ball / attack_step
    pcls_original = pcls.detach()
    pcls_adv = pcls.detach()
    for _ in range(step_size):
        pcls_adv.stop_gradient = False
        pcls_adv = pcls_adv + 0.01 * paddle.randn(pcls_adv.shape)
        loss_logits_kl = kldiv_criterion(logsoftmax(pcls_adv), softmax(pcls_original))
        grad = paddle.autograd.grad(loss_logits_kl, pcls_adv)[0]
        # avoid nan or inf if gradient is 0
        if grad.isnan().any():
            paddle.assign(0.001 * paddle.randn(grad.shape), grad)
        adv_img = adv_img.detach() + step_size * paddle.sign(grad.detach())
        eta = paddle.clip(adv_img - original_img, - epsilon_ball, epsilon_ball)
        adv_img = original_img + eta


def run(FLAGS, cfg):
    # build trainer
    trainer = Trainer(cfg, mode='test')

    # load weights
    trainer.load_weights(cfg.weights)

    # init depre
    depre_settings = {'ImPermute': {},
                      'DenormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                           'input_channel_axis': 2, 'is_scale': True},
                      'Resize': {'target_size': (404, 640, 3), 'keep_ratio': False, 'interp': 2},
                      'Encode': {}
                      }
    depreprocessor = OperatorCompose(depre_settings)
    draw_threshold = FLAGS.draw_threshold

    outs0, data0, datainfo0 = _image2outs(FLAGS.infer_dir, FLAGS.infer_img, trainer.model, cfg)
    outs_target, data_target, datainfo_target = _image2outs(FLAGS.infer_dir, FLAGS.target_img, trainer.model, cfg)

    # Supervised Attack. Use target_pcls to guide attack loss minimize process.
    target_pcls_list = get_pcls(trainer.model, outs_target['neck_feats'])

    attack_step = 10
    epsilon_ball = 100 / 255
    epsilon_stepsize = epsilon_ball / attack_step
    # enable gradients
    purturbation_shape = data0['image'].shape
    purturbation_shape = purturbation_shape[1:]
    purturbation = 0.01 * paddle.randn(purturbation_shape)
    # TODO: try to use Adam, cw attack.
    # optimizer = paddle.optimizer.Adam(parameters=[purturbation])
    for _ in range(attack_step):
        # optimizer.clear_grad()
        purturbation.stop_gradient = False
        data1 = copy.deepcopy(data0)
        data1['image'] = data1['image'] + purturbation
        outs1 = trainer.model(data1)

        pcls_list = get_pcls(trainer.model, outs1['neck_feats'])
        attack_loss = pcls_kldivloss(pcls_list, target_pcls_list)

        print(attack_loss)
        attack_loss.backward(retain_graph=True)
        # optimizer.step()

        purturbation_delta = - purturbation.grad
        # purturbation_delta = 0.1 * paddle.randn(purturbation_shape)
        normalized_purturbation_delta = paddle.sign(purturbation_delta)
        purturbation = epsilon_stepsize * normalized_purturbation_delta
        adv_img = data0['image'].detach() + purturbation.detach()
        purturbation = paddle.clip(adv_img - data0['image'], -epsilon_ball, epsilon_ball)


    adv_png = depreprocessor(adv_img[0])
    adv_png_path = 'output/adv_{imagename}.png'.format(imagename=os.path.basename(FLAGS.infer_img))
    cv2.imwrite(adv_png_path, adv_png)

    _draw_result_and_save(FLAGS.infer_img, outs0, data0, datainfo0, draw_threshold)
    _draw_result_and_save(FLAGS.target_img, outs_target, data_target, datainfo_target, draw_threshold)
    _draw_result_and_save(adv_png_path, outs1, data1, datainfo0, draw_threshold)


def _image2outs(infer_dir, infer_img, model, cfg):
    # get inference images
    images = get_test_images(infer_dir, infer_img)
    mode = 'test'
    dataset = cfg['{}Dataset'.format(mode.capitalize())]
    dataset.set_images(images*1)
    loader1 = create('TestReader')(dataset, 0)
    imid2path = dataset.get_imid2path()
    anno_file = dataset.get_anno()
    clsid2catid, catid2name = get_categories(cfg.metric, anno_file=anno_file)
    datainfo = {'imid2path': imid2path,
                'clsid2catid': clsid2catid,
                'catid2name': catid2name}

    model.eval()
    for step_id, data in enumerate(loader1):
        # forward
        outs = model(data)
    return outs, data, datainfo


def _draw_result_and_save(image_path, outs, data, datainfo, draw_threshold):
    clsid2catid = datainfo['clsid2catid']
    catid2name = datainfo['catid2name']
    for key in ['im_shape', 'scale_factor', 'im_id']:
        outs[key] = data[key]
    for key, value in outs.items():
        if hasattr(value, 'numpy'):
            outs[key] = value.numpy()

    batch_res = get_infer_results(outs, clsid2catid)
    bbox_num = outs['bbox_num']

    start = 0
    for i, im_id in enumerate(outs['im_id']):
        end = start + bbox_num[i]
        image = Image.open(image_path).convert('RGB')

        bbox_res = batch_res['bbox'][start:end] \
            if 'bbox' in batch_res else None
        mask_res = batch_res['mask'][start:end] \
            if 'mask' in batch_res else None
        segm_res = batch_res['segm'][start:end] \
            if 'segm' in batch_res else None
        keypoint_res = batch_res['keypoint'][start:end] \
            if 'keypoint' in batch_res else None
        image = visualize_results(
            image, bbox_res, mask_res, segm_res, keypoint_res,
            int(im_id), catid2name, draw_threshold)

        # save image with detection
        save_name = os.path.join('output/', 'out_' + os.path.basename(image_path))
        logger.info("Detection bbox results save in {}".format(
            save_name))
        image.save(save_name, quality=95)
        start = end


def test():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    merge_config(FLAGS.opt)

    place = paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    test()
