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
This module provides the attack method for HopSkipJumpAttack implement.
"""

from __future__ import absolute_import, division, print_function
from builtins import zip
from builtins import str
from builtins import range
import logging
from collections import Iterable

logger = logging.getLogger(__name__)

import numpy as np
from .base import Attack
import paddle


__all__ = [
    'HopSkipJumpAttack'
]


class HopSkipJumpAttack(Attack):
    """
    HopSkipJumpAttack
    """
    def __init__(self, model):
        """

        Args:
            model:
            support_targeted:
        """
        super(HopSkipJumpAttack, self).__init__(model)

    def _apply(self, adversary, steps=100):
        """

        Args:
            adversary:
            max_pixels:

        Returns:

        """
        if adversary.is_targeted_attack:
            raise ValueError(
                "This attack method doesn't support targeted attack!")

        min_, max_ = self.model.bounds

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.denormalized_original)
        perturbed = np.copy(adv_img)
        for i in range(steps):
            # TODO: duplicated lines?
            perturbed = paddle.to_tensor(perturbed, dtype='float32', place=self._device)
            perturbed = self.hsja(perturbed, adversary)
            perturbed = paddle.to_tensor(perturbed, dtype='float32', place=self._device)
            perturbed_normalized = self.input_preprocess(perturbed)

            adv_label = np.argmax(self.model.predict(perturbed_normalized))
            perturbed = self.safe_delete_batchsize_dimension(perturbed)
            perturbed_normalized = self.safe_delete_batchsize_dimension(perturbed_normalized)
            is_ok = adversary.try_accept_the_example(perturbed.numpy(),
                                                     perturbed_normalized.numpy(),
                                                     adv_label)
            if is_ok:
                return adversary

        return adversary

    def hsja(self,
        #model, 
        sample, 
        adversary,
        clip_max = 1.0, 
        clip_min = 0.0, 
        constraint = 'l2', 
        num_iterations = 1, 
        gamma = 1.0, 
        target_label = None, 
        target_image = None, 
        stepsize_search = 'geometric_progression', 
        max_num_evals = 1e4,
        init_num_evals = 100,
        verbose = True):
        """
        refer from: https://github.com/Jianbo-Lab/HSJA

        Main algorithm for HopSkipJumpAttack.

        Inputs:
        model: the object that has predict method. 

        predict outputs probability scores.

        clip_max: upper bound of the image.

        clip_min: lower bound of the image.

        constraint: choose between [l2, linf].

        num_iterations: number of iterations.

        gamma: used to set binary search threshold theta. The binary search 
        threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for 
        linf attack.

        target_label: integer or None for nontargeted attack.

        target_image: an array with the same size as sample, or None. 

        stepsize_search: choose between 'geometric_progression', 'grid_search'.

        max_num_evals: maximum number of evaluations for estimating gradient (for each iteration). 
        This is not the total number of model evaluations for the entire algorithm, you need to 
        set a counter of model evaluations by yourself to get that. To increase the total number 
        of model evaluations, set a larger num_iterations. 

        init_num_evals: initial number of evaluations for estimating gradient.

        Output:
        perturbed image.
        
        """
        # Set parameters
        #original_label = paddle.argmax(self.model.predict(sample))
        original_label = adversary.original_label
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
        perturbed = self.initialize(sample, params)
        

        # Project the initialization to the boundary.
        perturbed, dist_post_update = self.binary_search_batch(sample, 
            np.expand_dims(perturbed, 0), 
            #model, 
            params)
        dist = self.compute_distance(perturbed, sample, constraint)

        for j in np.arange(params['num_iterations']):
            params['cur_iter'] = j + 1

            # Choose delta.
            delta = self.select_delta(params, dist_post_update)

            # Choose number of evaluations.
            num_evals = int(params['init_num_evals'] * np.sqrt(j + 1))
            num_evals = int(min([num_evals, params['max_num_evals']]))

            # approximate gradient.
            gradf = self.approximate_gradient(perturbed, num_evals, delta, params)
            if params['constraint'] == 'linf':
                update = np.sign(gradf)
            else:
                update = gradf

            # search step size.
            if params['stepsize_search'] == 'geometric_progression':
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(perturbed, 
                    update, dist, params)

                # Update the sample. 
                perturbed = self.clip_image(perturbed + epsilon * update, 
                    clip_min, clip_max)

                # Binary search to return to the boundary. 
                perturbed, dist_post_update = self.binary_search_batch(sample, 
                    perturbed[None], params)

            elif params['stepsize_search'] == 'grid_search':
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint = True) * dist
                epsilons_shape = [20] + len(params['shape']) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = self.clip_image(perturbeds, params['clip_min'], params['clip_max'])
                idx_perturbed = self.decision_function(perturbeds, params)

                if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update = self.binary_search_batch(sample, 
                                            perturbeds[idx_perturbed], params)

            # compute new distance.
            dist = self.compute_distance(perturbed, sample, constraint)
            if verbose:
                print('iteration: {:d}, {:s} distance {:.4E}'.format(j + 1, constraint, dist))

        return perturbed

    def decision_function(self, images, params):
        """
        Decision function output 1 on the desired side of the boundary,
        0 otherwise.
        """
        images = self.clip_image(images, params['clip_min'], params['clip_max'])
        image = paddle.to_tensor(images[0].astype('float32'))
        prob = self.model.predict(images)
        if params['target_label'] is None:
            return np.argmax(prob, axis = 1) != params['original_label'] 
        else:
            return np.argmax(prob, axis = 1) == params['target_label']

    def clip_image(self, image, clip_min, clip_max):
        """
        Clip an image, or an image batch, with upper and lower threshold.
        """
        return np.minimum(np.maximum(clip_min, image), clip_max) 

    def compute_distance(self, x_ori, x_pert, constraint = 'l2'):
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

    def approximate_gradient(self, sample, num_evals, delta, params):
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
            rv = np.random.uniform(low = -1, high = 1, size = noise_shape)

        rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1, 2, 3), keepdims = True))
        perturbed = sample + delta * rv
        perturbed = self.clip_image(perturbed, clip_min, clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions = self.decision_function(perturbed, params)
        decision_shape = [len(decisions)] + [1] * len(params['shape'])
        fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0
        if isinstance(fval, paddle.Tensor):
            fval = fval.numpy()

        # Baseline subtraction (when fval differs)
        if np.mean(fval) == 1.0: # label changes. 
            gradf = np.mean(rv, axis = 0)
        elif np.mean(fval) == -1.0: # label not change.
            gradf = - np.mean(rv, axis = 0)
        else:
            fval -= np.mean(fval)
            gradf = np.mean(fval * rv, axis = 0) 

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)

        return gradf

    def project(self, original_image, perturbed_images, alphas, params):
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
            out_images = self.clip_image(
                perturbed_images, 
                original_image - alphas, 
                original_image + alphas
                )
            return out_images

    def binary_search_batch(self, original_image, perturbed_images, params):
        """ Binary search to approach the boundar. """

        # Compute distance between each of perturbed image and original image.
        dists_post_update = np.array([
            self.compute_distance(
            original_image, 
            perturbed_image, 
            params['constraint']
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
            mid_images = self.project(original_image, perturbed_images, mids, params)

            # Update highs and lows based on model decisions.
            decisions = self.decision_function(mid_images, params)
            print("==decisions: ", decisions, type(decisions))
            #Tensor(shape=[1], dtype=bool, place=CUDAPlace(0), stop_gradient=False,
                   #[False]) <class 'paddle.Tensor'>

            if isinstance(decisions, paddle.Tensor):
                decisions = decisions.numpy()
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        out_images = self.project(original_image, perturbed_images, highs, params)

        # Compute distance of the output image to select the best choice. 
        # (only used when stepsize_search is grid_search.)
        dists = np.array([
        self.compute_distance(
            original_image, 
            out_image, 
            params['constraint']
        ) 
        for out_image in out_images])
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist

    def initialize(self, sample, params):
        """ 
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0

        if params['target_image'] is None:
            # Find a misclassified random noise.
            while True:
                random_noise = np.random.uniform(params['clip_min'], 
                params['clip_max'], size = params['shape'])
                success = self.decision_function(random_noise[None], params)
                num_evals += 1
                if success:
                    break
                assert num_evals < 1e4, "Initialization failed! "
                "Use a misclassified image as `target_image`" 

            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample.numpy() + mid * random_noise 
                success = self.decision_function(blended[None], params)
                if success:
                    high = mid
                else:
                    low = mid

            initialization = (1 - high) * sample.numpy() + high * random_noise 

        else:
            initialization = params['target_image']

        return initialization

    def geometric_progression_for_stepsize(self, x, update, dist, params):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching 
        the desired side of the boundary,
        """
        epsilon = dist / np.sqrt(params['cur_iter']) 

        def phi(epsilon):
            new = x + epsilon * update
            success = self.decision_function(new[None], params)
            return success

        while not phi(epsilon):
            epsilon /= 2.0

        return epsilon

    def select_delta(self, params, dist_post_update):
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
