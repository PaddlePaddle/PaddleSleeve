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
The HopSkipJumpAttack black aattack implementation on object detection.
Contains:
* Initialize a inference pictures.
* Generate adversarial perturbation by estimating the gradient direction using binary information at the decision boundary.
Author: tianweijuan
"""

from __future__ import absolute_import
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))
import warnings
warnings.filterwarnings('ignore')
import glob
import cv2
import math
from PIL import Image

import xmltodict
import numpy
import paddle
import paddle.nn.functional as F
from ppdet.core.workspace import create
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.data.source.category import get_categories
from ppdet.metrics import get_infer_results
from ppdet.utils.logger import setup_logger
import numpy as np

from depreprocess.operator_composer import OperatorCompose
from past.utils import old_div


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
        "--draw_threshold",
        type=float,
        default=0.6,
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
        '--max_steps', 
        type=int, 
        default=500, 
        help="maximum steps")
    parser.add_argument(
        '--norm', 
        type=str, 
        default='l2', 
        help="specify the norm of the attack, choose one from 'l2'or 'linf'")
    
    parser.add_argument(
        '--sim_label', 
        nargs='+',
        type=int,
        help="specify the similar label of the object label, the first is groundtruth label")
        
    parser.add_argument(
        '--target_label', 
        type=int, 
        default=None, 
        help="specify the target label if targeted attack")
        
    parser.add_argument(
        '--target_image', 
        type=str, 
        default='./dataloader/hydrant1.png', 
        help="specify the target image for targeted attack if target label exists")

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


def decision_function(model, images, params, datainfo, data0, sim_label):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    images = clip_image(images, params['clip_min'], params['clip_max'])
    images = paddle.to_tensor(images[0].astype('float32'))
    c, _, _ = images.shape
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    for j in range(c):
        images[j] = old_div(images[j] - mean[j],
                            std[j])
    images = paddle.unsqueeze(images, axis=0)

    data0["image"] = images
    prob_label, score, score_orig = ext_score(model(data0), data0, datainfo, sim_label)
    if score_orig > 0.3:
        prob_label = sim_label[0]
    if params['target_label'] is None:
        if score_orig == 0. and prob_label == sim_label[0]:
            return np.array([score_orig == 0.])
        return np.array([prob_label != params['original_label'] and prob_label not in sim_label])
    else:
        return np.array([prob_label == params['target_label']])

def clip_image(image, clip_min, clip_max):
    """
    Clip an image, or an image batch, with upper and lower threshold.
    """
    return np.minimum(np.maximum(clip_min, image), clip_max)

def compute_distance(x_ori, x_pert, constraint = 'l2'):
    """
    args:x_ori, x_pert, constraint
    return: distance
    """
    # Compute the distance between two images.
    if isinstance(x_ori, paddle.Tensor):
        x_ori = x_ori.numpy()
    if isinstance(x_pert, paddle.Tensor):
        x_pert = x_pert.numpy()

    if constraint == 'l2':
        return np.linalg.norm(x_ori - x_pert)
    elif constraint == 'linf':
        return np.max(abs(x_ori - x_pert))

def approximate_gradient(model, sample, num_evals, delta, params, datainfo, data0, sim_label):
    """
    approximate gradient
    args:model, sample, num_evals, delta, params
    return: grad
    """
    clip_max, clip_min = params['clip_max'], params['clip_min']

    # Generate random vectors.
    noise_shape = [num_evals] + list(params['shape'])
    if params['constraint'] == 'l2':
        rv = np.random.randn(*noise_shape)
    elif params['constraint'] == 'linf':
        rv = np.random.uniform(low=-1, high=1, size=noise_shape)

    rv = rv / np.sqrt(np.sum(rv ** 2, axis=(1, 2, 3), keepdims=True))
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(model, perturbed, params, datainfo, data0, sim_label) * 1


    decision_shape = [len(decisions)] + [1] * len(params['shape'])
    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0
    if isinstance(fval, paddle.Tensor):
        fval = fval.numpy()

    # Baseline subtraction (when fval differs)
    if np.mean(fval) == 1.0:  # label changes.
        gradf = np.mean(rv, axis=0)
    elif np.mean(fval) == -1.0:  # label not change.
        gradf = - np.mean(rv, axis=0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * rv, axis=0)

        # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)
    return gradf

def initialize(model, sample, params, datainfo, data0, orig, sim_label):
    """
    Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
    """
    success = 0
    num_evals = 0
    if params['target_label'] is None:
        # Find a misclassified random noise.
        while True:
            random_noise = np.random.uniform(params['clip_min'],
                                             params['clip_max'],
                                             size=params['shape'])
            success = decision_function(model, random_noise[None], params, datainfo, data0, sim_label) # 寻找导致分类错误的噪声
            num_evals += 1
            if success: #and dist < 700:
                break
            assert num_evals < 1e4, "Initialization failed! "
            "Use a misclassified image as `target_image`"
            "Use a misclassified image as `target_image`"

            # Binary search to minimize l2 distance to original image.

        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid) * sample + mid * random_noise
            success = decision_function(model, blended[None], params, datainfo, data0, sim_label)
            if success:
                high = mid
            else:
                low = mid

        initialization = (1 - high) * sample + high * random_noise

    else:
        initialization = params['target_image']

    return initialization

def geometric_progression_for_stepsize(model, x, update, dist, params, datainfo, data0, sim_label):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching
    the desired side of the boundary,
    """
    epsilon = dist / np.sqrt(params['cur_iter'])

    def phi(epsilon):
        new = x + epsilon * update
        success = decision_function(model, new[None], params, datainfo, data0, sim_label)
        return success

    while not phi(epsilon):
        epsilon /= 2.0

    return epsilon

def select_delta(params, dist_post_update):
    """
    Choose the delta at the scale of distance
    between x and perturbed sample.
    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist_post_update
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update

    return delta

def project(original_image, perturbed_images, alphas, params):
    """
    project
    """
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)

    if isinstance(original_image, paddle.Tensor):
        original_image = original_image.numpy()
    if isinstance(perturbed_images, paddle.Tensor):
        perturbed_images = perturbed_images.numpy()

    if params['constraint'] == 'l2':
        return (1 - alphas) * original_image + alphas * perturbed_images
    elif params['constraint'] == 'linf':
        out_images = clip_image(
            perturbed_images,
            original_image - alphas,
            original_image + alphas
        )
        return out_images

def binary_search_batch(model, original_image, perturbed_images, params, datainfo, data0, sim_label):
    """ Binary search to approach the boundar. """
    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
            compute_distance(
            original_image,
            perturbed_image[0],
            "l2"#params['constraint']
        )
        for perturbed_image in perturbed_images])
    # Choose upper thresholds in binary searchs based on constraint.
    if params['constraint'] == 'linf':
        highs = dists_post_update
        # Stopping criteria.
        thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
    else:
        highs = np.ones(len(perturbed_images))
        thresholds = params['theta']
    lows = np.zeros(len(perturbed_images))
    # Call recursive function.
    while np.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)
        # Update highs and lows based on model decisions.
        decisions = decision_function(model, mid_images, params, datainfo, data0, sim_label)
        if isinstance(decisions, paddle.Tensor):
            decisions = decisions.numpy()
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)
    out_images = project(original_image, perturbed_images, highs, params)
    # Compute distance of the output image to select the best choice.
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
            compute_distance(
            original_image,
            out_image[0],
            "l2"#params['constraint']
        )
        for out_image in out_images])
    idx = np.argmin(dists)
    dist = dists_post_update[idx]
    out_image = out_images[idx]
    return out_image, dist
def hsja(
         model,
         original_label,
         sample,
         datainfo,
         data0,
         orig,
         sim_label,
         clip_max=1.0,
         clip_min=0.0,
         constraint='l2',
         num_iterations=1,
         gamma=1.0,
         target_label=None,
         target_image=None,
         stepsize_search='geometric_progression',
         max_num_evals=1e4,
         init_num_evals=100,
         verbose=True):
    params = {'clip_max': clip_max, 'clip_min': clip_min,
              'shape': sample.shape,
              'original_label': original_label,
              'target_label': target_label,
              'target_image': target_image,
              'constraint': constraint,
              'num_iterations': num_iterations,
              'gamma': gamma,
              'd': int(np.prod(sample.shape)),
              'stepsize_search': stepsize_search,
              'max_num_evals': max_num_evals,
              'init_num_evals': init_num_evals,
              'verbose': verbose,
              }
    # Set binary search threshold.
    if params['constraint'] == 'l2':
        params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])
    else:
        params['theta'] = params['gamma'] / (params['d'] ** 2)
    # Initialize.
    perturbed = initialize(model, sample, params, datainfo, data0, orig, sim_label)
 
    # Project the initialization to the boundary.
    
    perturbed, dist_post_update = binary_search_batch(model, orig, # sample
                                                           np.expand_dims(perturbed, 0),
                                                           params,
                                                           datainfo,
                                                           data0,
                                                           sim_label)
    dist = compute_distance(orig, perturbed, constraint) #sample
    for j in np.arange(params['num_iterations']):
        params['cur_iter'] = j + 1
        # Choose delta.
        delta = select_delta(params, dist_post_update)
        # Choose number of evaluations.
        num_evals = int(params['init_num_evals'] * np.sqrt(j + 1))
        num_evals = int(min([num_evals, params['max_num_evals']]))
        # approximate gradient.
        gradf = approximate_gradient(model, perturbed, num_evals, delta, params, datainfo, data0, sim_label)
        if params['constraint'] == 'linf':
            update = np.sign(gradf)
        else:
            update = gradf
        # search step size.
        if params['stepsize_search'] == 'geometric_progression':
            # find step size.
            epsilon = geometric_progression_for_stepsize(model, perturbed,
                                                              update, dist, params, datainfo, data0, sim_label)
            # Update the sample.
            perturbed = clip_image(perturbed + epsilon * update,
                                        clip_min, clip_max)
            # Binary search to return to the boundary.
            perturbed, dist_post_update = binary_search_batch(model, orig, # sample
                                                                   perturbed[None], params,
                                                                   datainfo, data0, sim_label)
        elif params['stepsize_search'] == 'grid_search':
            # Grid search for stepsize.
            epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
            epsilons_shape = [20] + len(params['shape']) * [1]
            perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
            perturbeds = clip_image(perturbeds, params['clip_min'], params['clip_max'])
            idx_perturbed = decision_function(perturbeds, params)
            if np.sum(idx_perturbed) > 0:
                # Select the perturbation that yields the minimum distance # after binary search.
                perturbed, dist_post_update = binary_search_batch(model, orig, # sample
                                                                       perturbeds[idx_perturbed], params,
                                                                       datainfo, data0, sim_label)
        # compute new distance.
        dist = compute_distance(orig, perturbed, constraint) # sample
        if verbose:
            print('iteration: {:d}, {:s} distance {:.4E}'.format(j + 1, constraint, dist))
    return perturbed, dist
def run(FLAGS, cfg):
    """
    construct input data and call the AttackNet to achieve the adversarial patch learning.
    Args:
        FLAGS(dict): configure parameters
        cfg(str): attacked model configs
    """

    # init depre
    img_path =  FLAGS.infer_img
    img = cv2.imread(img_path)
    H, W, C  = img.shape

    depre_settings = {'ImPermute': {},
                      'DenormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                           'input_channel_axis': 2, 'is_scale': True},
                      'Resize': {'target_size': (H, W, C), 'keep_ratio': False, 'interp': 2}, #638, 850, 864, 1152,
                      'Encode': {}
                      }
    
    depreprocessor = OperatorCompose(depre_settings)
    draw_threshold = FLAGS.draw_threshold
    
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)  
    
    data0, datainfo0  = _image2outs(FLAGS.infer_dir, FLAGS.infer_img, cfg)
    _, C, H, W = data0["image"].shape
    if FLAGS.target_label:
        target_image = FLAGS.target_image # None
        target_image_tensor, _ = _image2outs(FLAGS.infer_dir, target_image, cfg)
        target_image = np.squeeze(target_image_tensor["image"])
        for i in range(C):
            target_image[i] = target_image[i] * depre_settings["DenormalizeImage"]["std"][i] + \
                         depre_settings["DenormalizeImage"]["mean"][i]


    else:
        target_image = None
        
    trainer.model.eval()
    
    orig_label, score, score_orig = ext_score(trainer.model(data0), data0, datainfo0, FLAGS.sim_label)
    if score_orig > 0.3:
        orig_label = FLAGS.sim_label[0]

    inputs = np.squeeze(data0["image"])

    adv_img =  np.zeros(inputs.shape)
    C, H, W = inputs.shape

    for i in range(C):
        adv_img[i] = inputs[i] * depre_settings["DenormalizeImage"]["std"][i] +  depre_settings["DenormalizeImage"]["mean"][i]

    for i in range(FLAGS.max_steps):
        print("step=======", i)
        perturbed = np.copy(adv_img)
        perturbed = np.clip(perturbed, 0., 1.)
        # perturbed = paddle.to_tensor(perturbed, dtype='float32', place=paddle.CUDAPlace(0))

        perturbed, dist = hsja(trainer.model, orig_label, perturbed, datainfo0, data0, adv_img, FLAGS.sim_label, constraint = FLAGS.norm,
                               num_iterations = 1, target_label = FLAGS.target_label, target_image = target_image)
        perturbed = paddle.to_tensor(perturbed, dtype='float32', place=paddle.CUDAPlace(0))
        perturbed_normalized = perturbed.clone()
        for j in range(C):
            perturbed_normalized[j] = old_div(perturbed_normalized[j] - depre_settings["DenormalizeImage"]["mean"][j], depre_settings["DenormalizeImage"]["std"][j])
        perturbed_normalized = paddle.unsqueeze(perturbed_normalized, axis=0)
        data0["image"] = perturbed_normalized

        adv_label, score, orig = ext_score(trainer.model(data0), data0, datainfo0, FLAGS.sim_label)
        if orig > 0.3: 
            adv_label = FLAGS.sim_label[0]
        print(adv_label, score, orig)

        dist1 = compute_distance(inputs, perturbed_normalized[0], constraint='l2')
        print("dist1=======", dist1)

        # if dist1 > 550: # fatser-rcnn, detr: 2200, yolov3:450, ppyolo:630, ssd: 250
        #     continue
        if FLAGS.target_label is None:
            if adv_label not in FLAGS.sim_label: 
                 data_adv = depreprocessor(perturbed_normalized[0].detach())
                 cv2.imwrite("./adv_detr.png", data_adv)
                 # 添加下面代码的原因是，图像输出和原图大小有一定的区别，而扰动是添加在模型输入图像上的，因此，扰动还原到原图上时，
                 # 扰动数据的分布会发生一定的变化，导致攻击的效果发生变化，因此为了确保攻击的成功率，对于初步攻击成功的样本进行检测判断，
                 # 确定检测结果失效，则保存攻击样本，否则继续执行攻击算法。

                 data1, _ = _image2outs(FLAGS.infer_dir, "./adv_detr.png", cfg)
                 trainer.model.eval()
                 orig_label1, score1, score_orig1 = ext_score(trainer.model(data1), data1, datainfo0, FLAGS.sim_label)
                 print("orig_label======", orig_label1, score_orig1, score1)
                 if score_orig1 > 0.5 or orig_label1 in FLAGS.sim_label: #or orig_label1 != 1:
                     continue
                 else:
                     break
        else:
            if adv_label == FLAGS.target_label:
                print("succes==============success")
                data_adv = depreprocessor(perturbed_normalized[0].detach())
                cv2.imwrite("./adv_detr_target.png", data_adv)

                data1, _ = _image2outs(FLAGS.infer_dir, "./adv_detr_target.png", cfg)
                trainer.model.eval()
                orig_label1, score1, score_orig1 = ext_score(trainer.model(data1), data1, datainfo0, FLAGS.sim_label)
                print(orig_label1, score1, score_orig1)
                # if orig_label1 == target_label and score1 > 0.3 and score_orig1 < 0.5:
                if orig_label1 == FLAGS.target_label and score1 > 0.3 and score_orig1 < 0.3:
                    break


def _image2outs(infer_dir, infer_img, cfg):
    """
    construct the single input data for the post data process.
    Args:
        infer_dir(str): input data path 
        infer_img(str): input data filename
        cfg(dict): attacked model definition
    """
   
    mode = 'Test'
    dataset = cfg['{}Dataset'.format(mode)] = create(
    '{}Dataset'.format(mode))()
    
    images = get_test_images(infer_dir, infer_img)
    dataset.set_images(images)

    loader = create('TestReader')(dataset, 0)
    imid2path = dataset.get_imid2path
    anno_file = dataset.get_anno()
    clsid2catid, catid2name = get_categories(cfg.metric, anno_file=anno_file)
    datainfo = {'imid2path': imid2path,
                'clsid2catid': clsid2catid,
                'catid2name': catid2name}
    # loader have 1 images, step_id=0, data contain bbox, bbox_num, neck_feats
    for step_id, data in enumerate(loader):
        break      
    return data, datainfo 
   

def ext_score(outs, data, datainfo, sim_label):
    """
    extract the detection score of the learned adversarial sample.
    Args:
        outs(dict): output of the learned adversarial sample 
        data(dict): the learned adversarial sample
        datainfo(dict): data information of the specific dataset
        label_id(int): the original target class id 
    """

    clsid2catid = datainfo['clsid2catid']
    catid2name = datainfo['catid2name']
    for key in ['im_shape', 'scale_factor', 'im_id']:
        outs[key] = data[key]
    for key, value in outs.items():
        if hasattr(value, 'numpy'):
            outs[key] = value.numpy()
    orig_label = sim_label[0]
    batch_res = get_infer_results(outs, clsid2catid)
    score_orig = 0.
    bbox = batch_res["bbox"]
    for box in bbox:
        if box["category_id"] == orig_label:
            score_orig = box["score"]
    start = 0
    max_score = 0
    max_label = orig_label
    bbox_num = outs['bbox_num']
    for i, im_id in enumerate(outs['im_id']):
        end = start + bbox_num[i]
        bbox_res = batch_res['bbox'][start:end]
        for dt in numpy.array(bbox_res):
            catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
            if catid in sim_label:
                if score_orig < score:
                    score_orig = score
            if score > max_score:
                max_score = score
                max_label = catid
    return max_label, max_score, score_orig

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
