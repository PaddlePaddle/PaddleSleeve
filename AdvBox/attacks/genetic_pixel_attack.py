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
This module provides the attack method for SinglePixelAttack & LocalSearchAttack's implement.
"""
from __future__ import division

from builtins import zip
from builtins import str
from builtins import range
import logging
from collections import Iterable

logger = logging.getLogger(__name__)

import numpy as np
from .base import Attack
import paddle
from scipy.special import softmax

__all__ = [
    'GeneticPixelAttack'
]


# Simple Black-Box Adversarial Perturbations for Deep Networks
# 随机在图像中选择max_pixels个点 在多个信道中同时进行修改，修改范围通常为0-255
class GeneticPixelAttack(Attack):
    """
    GeneticPixelAttack
    """

    def __init__(self, model, target=-1, max_pixels=40,
                 population=15, mutation_rate=0.05, max_gen=5000, temp=100, threshold=200):
        """

        Args:
            model: An instance of a paddle model to be attacked.
            target(int): Target class, -1 if untargeted
            max_pixels(int): Max number of pixels allowed to change
            population(int): The max number of candidates in any generation
            mutation_rate(float): The probability that mutations happen, this should be in range (0,1)
            max_steps(int): Max number of iterations before return fail
            temp(float): Controls how likely an unfavored candidate will be selected as parent

        """
        super(GeneticPixelAttack, self).__init__(model, )

        assert type(max_pixels) == int and max_pixels >= 0, "Invalid argument: max_pixels should be a positive integer"
        assert type(population) == int and population >= 0, "Invalid argument: Population should be a positive integer"
        assert 0 < mutation_rate < 1, "Invalid argument: Mutation rate should be within (0,1)"
        assert type(max_gen) == int and max_gen > 0, "Invalid argument: Max generations should be a positive integer"

        self._target = target
        self._max_pixels = max_pixels
        self._population = population
        self._mutation_rate = mutation_rate
        self._max_gen = max_gen
        self._temp = temp
        self._threshold = threshold
        # initial global optimum fitness value
        self._best_fit = -np.inf
        # count times of no progress
        self._plateau_times = 0
        self.lb, self.ub = self.model.bounds

    # 如果输入的原始数据，isPreprocessed为False，如果驶入的图像数据被归一化了，设置为True
    def _apply(self, adversary):
        """

        Args:
            adversary:
            max_pixels(int): Max number of pixels allowed  changing

        Returns:

        """
        # if adversary.is_targeted_attack:
        #     raise ValueError(
        #         "This attack method doesn't support targeted attack!")

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.denormalized_original)

        axes = [i for i in range(adversary.original.ndim) if i != self.model.input_channel_axis]

        # 输入的图像必须具有长和宽属性
        assert len(axes) == 2

        h, w = adv_img.shape[1:]

        # =======================================================================================
        # Improvement on pixel attack using genetic approach

        # Create the initial population
        adv_img = np.moveaxis(adv_img, self.model.input_channel_axis, 0)
        adv_img = adv_img.reshape(-1, h * w)
        people, dirtmap = self._generate_batch(adv_img, self._population, self._max_pixels)

        cur_gen = 0
        cur_temp = self._temp
        temp_step = 0
        scale = 0
        threshold = self._threshold
        target = self._target

        # Main evolution loop
        while (cur_gen < self._max_gen):
            people_norm = [self.safe_delete_batchsize_dimension(
                           self.input_preprocess(
                           paddle.to_tensor(
                           individual.reshape(adversary.original.shape),
                           dtype='float32', place=self._device))).numpy()
                           for individual in people]
            people_norm = np.stack(people_norm, axis=0)
            people_norm = paddle.to_tensor(people_norm, dtype='float32', place=self._device)

            labels = self.model.predict(people_norm)
            tag = np.argmax(labels, axis=1)
            if target != -1:
                success = np.argwhere(tag == target)
            else:
                success = np.argwhere(tag != adversary.original_label)


            if success.size != 0:
                success = int(success[0])
                success_norm = self.safe_delete_batchsize_dimension(people_norm[success, :]).numpy()
                success_ori = people[success, :].reshape(adversary.original.shape)
                is_ok = adversary.try_accept_the_example(success_ori,
                                                         success_norm,
                                                         tag[success])
                if is_ok:
                    return adversary

            # If no successful adversary
            labels = labels - labels[:, adversary.original_label:adversary.original_label+1]
            labels[:,adversary.original_label] = -np.inf
                
            if target != -1:
                score = labels[:, target]
            else:
                score = np.max(labels, axis=1)
#            score = -labels[:, adversary.original_label]
            elite = np.argmax(score)
            cur_best = score[elite]
            if cur_gen == 0:
                scale = 1 + np.abs(score[elite]) // 10
            if cur_gen % 500 == 0:
                logging.info("Current Step {0}, Best {1}".format(score[elite-1],cur_best))
            if cur_best > self._best_fit:
                self._best_fit = cur_best
                self._plateau_times = 0
                threshold = self._threshold
            else:
                self._plateau_times += 1

            if self._plateau_times > threshold:
                kids, dirtmap_kids = self._generate_batch(np.copy(people[elite]), self._population-1, self._max_pixels // 2)
                dirtmap_kids = [entry + dirtmap[elite] for entry in dirtmap_kids]
#                # kids, dirtmap_kids = self._playground(kids, dirtmap_kids, 50, self._population-1, adversary.denormalized_original, adversary.original_label)
#                people = np.concatenate((kids, people[elite:elite + 1, :]))
#                dirtmap_kids.append(dirtmap[elite])
#                dirtmap = dirtmap_kids
#                cur_gen += 1
#                temp_step = 0
#                self._temp = min(1e4, self._temp * 2)
                self._plateau_times = 0
                threshold =max(500, threshold+100)
            else:
                cur_temp = self._temp * np.power(0.5, cur_gen / self._max_gen)
    
                prob = softmax(score / scale / cur_temp)
                if cur_gen % 500 == 0:
                    print('prob', prob[elite], cur_temp)
                select_args = np.arange(self._population)
                kids = []
                dirtmap_kids = []
                for i in range(self._population - 1):
                    p1, p2 = np.random.choice(a=select_args, size=2, p=prob)
    #                ngene = int(score[p2] / (score[p2]+score[p1]) * len(dirtmap[p1])) + 1
                    ngene = len(dirtmap[p1]) // 2
                    dirtybit1 = dirtmap[p1][:ngene]
                    dirtybit2 = dirtmap[p2][-(self._max_pixels - ngene):]
                    dirtybit1 = dirtybit1 if type(dirtybit1) == list else [dirtybit1]
                    dirtybit2 = dirtybit2 if type(dirtybit2) == list else [dirtybit2]
                    kid = np.copy(adv_img)
                    kid[:, dirtybit1] = people[p1:p1 + 1, :, dirtybit1]
                    kid[:, dirtybit2] = people[p2:p2 + 1, :, dirtybit2]
                    dirtmap_kids.append(list(set(dirtybit2 + dirtybit1)))
                    kids.append(kid)
    
                kids = np.stack(kids)
                kids, dirtmap_kids = self._mutation(kids, dirtmap_kids, self._mutation_rate)
    
            # For each child, check whether too many pixels have been changed
            for i in range(kids.shape[0]):
                if len(dirtmap_kids[i]) > self._max_pixels:
                    fix_pixel = np.random.choice(dirtmap_kids[i], len(dirtmap_kids[i]) - self._max_pixels, replace=False)
                    kids[i:i + 1, :, fix_pixel] = adv_img[:, fix_pixel]
                    dirtmap_kids[i] = [x for x in dirtmap_kids[i] if x not in fix_pixel]

            people = np.concatenate((kids, people[elite:elite + 1, :]))
            dirtmap_kids.append(dirtmap[elite])
            dirtmap = dirtmap_kids
            cur_gen += 1
            temp_step += 1

        # =======================================================================================

        # print("w={0},h={1}".format(w,h))
        # max_pixels为攻击点的最多个数 从原始图像中随机选择max_pixels个进行攻击

        # pixels = np.random.permutation(h * w)
        # pixels = pixels[:max_pixels]
        #
        # for i, pixel in enumerate(pixels):
        #     x = pixel % w
        #     y = pixel // w
        #
        #     location = [x, y]
        #     if i % 50 == 0:
        #         logging.info("Attack location x={0} y={1}".format(x, y))
        #
        #     location.insert(self.model.input_channel_axis, slice(None))
        #     location = tuple(location)
        #
        #     # TODO: add differential evolution
        #     for value in [min_, max_]:
        #         perturbed = np.copy(adv_img)
        #         # 针对图像的每个信道的点[x,y]同时进行修改
        #         perturbed[location] = value
        #
        #         perturbed = paddle.to_tensor(perturbed, dtype='float32', place=self._device)
        #
        #         perturbed_normalized = self.input_preprocess(perturbed)
        #         adv_label = np.argmax(self.model.predict(perturbed_normalized))
        #
        #         perturbed = self.safe_delete_batchsize_dimension(perturbed)
        #         perturbed_normalized = self.safe_delete_batchsize_dimension(perturbed_normalized)
        #         is_ok = adversary.try_accept_the_example(perturbed.numpy(),
        #                                                  perturbed_normalized.numpy(),
        #                                                  adv_label)
        #         if is_ok:
        #             return adversary

        return adversary

    def _mutation(self, kids, dirtmap, rate):
        """

        Args:
            kids(numpy.ndarray): All new children to be mutated
            dirtmap(list): Pixels that are already mutated
            rate: mutation rate

        Returns:
            numpy.ndarray, children after mutation

        """
        assert kids.ndim == 3, "Invalid argument: ndim not correct"

        # changed_pixel = np.argwhere(np.maximum(rate - np.random.random((children.shape[0], num_pixel)), 0) > 0)
        _bound = self.ub - self.lb
        for i, kid in enumerate(kids):
            # Perturb more pixels if possible

            avail = self._max_pixels - len(dirtmap[i])
            avail = avail+1 if np.random.random() < rate else avail
            if avail > 0:
                pert = np.random.choice(range(kid.shape[1]),avail,replace=False)
                new_val = np.ones((kid.shape[0],1)) * np.random.choice([self.lb, self.ub], avail) 
                kid[:,pert] = new_val
                dirtmap[i] = list(set(dirtmap[i] + pert.tolist())) 

            # Add small pertubations to dirty bits 
            _max = max(1, rate * 400) # self._plateau_times)
            amp = 0.01 * _bound * np.ones((kid.shape[0],1)) * np.random.randint(-_max, _max, len(dirtmap[i])) 
            kid[:,dirtmap[i]] =np.clip(kid[:,dirtmap[i]] + amp, self.lb, self.ub)
        return kids, dirtmap




    def _generate_batch(self, img, pop, npixels):
        """

        Args:
            img(numpy.ndarray): Original img based on which the batch is generated
            pop(int): batchsize

        Returns:
            numpy.ndarray, children after mutation
            list: Bit map that record the dirty bit

        """

        people = np.repeat(img[np.newaxis, :], pop, axis=0)
        # The map recording the mutated pixels for entire population

        # # Add initial perturb to population
        # people, dirtmap = self._mutation(people, dirtmap, self._mutation_rate)
        pert_args = np.random.randint(0, img.shape[1], (pop,npixels))
        for i in range(pop):
            temp = np.random.random(npixels) * (self.ub - self.lb) + self.lb
            temp = np.ones((img.shape[0],1)) * temp
            people[i,:,pert_args[i]] = np.transpose(temp)
        dirtmap = pert_args.tolist()

        return people, dirtmap

 


    def _playground(self, people, dirtmap, nstep, pop, adv_img, label):
        """

        Args:

        Returns:
            numpy.ndarray, children after mutation
            list: Bit map that record the dirty bit

        """

        c, h, w = adv_img.shape
        adv_img = adv_img.reshape((c, h*w))
        cur_temp = 1
        best = -np.inf
        # Main evolution loop
        for step in range(nstep):
            people_norm = [self.safe_delete_batchsize_dimension(
                           self.input_preprocess(
                           paddle.to_tensor(
                           individual.reshape((c,h,w)),
                           dtype='float32', place=self._device))).numpy()
                           for individual in people]
            people_norm = np.stack(people_norm, axis=0)
            people_norm = paddle.to_tensor(people_norm, dtype='float32', place=self._device)

            labels = self.model.predict(people_norm)
            tag = np.argmax(labels, axis=1)
            success = np.argwhere(tag != label)


            if success.size != 0:
                return people, dirtmap

            # If no successful adversary
            labels = labels - labels[:, label:label+1]
            labels[:, label] = -np.inf
            score = np.max(labels, axis=1)
#            score = -labels[:, label]
            elite = np.argmax(score)
            cur_best = score[elite]

            if cur_best > best:
                best = cur_best

            prob = softmax(score / cur_temp)
            select_args = np.arange(pop)
            kids = []
            dirtmap_kids = []
            for i in range(pop - 1):
                p1, p2 = np.random.choice(a=select_args, size=2, p=prob)
                ngene = int(score[p2] / (score[p2]+score[p1]) * len(dirtmap[p1])) + 1
                # ngene = len(dirtmap[p1]) // 2
                dirtybit1 = dirtmap[p1][:ngene]
                dirtybit2 = dirtmap[p2][-(self._max_pixels - ngene):]
                dirtybit1 = dirtybit1 if type(dirtybit1) == list else [dirtybit1]
                dirtybit2 = dirtybit2 if type(dirtybit2) == list else [dirtybit2]
                kid = np.copy(adv_img)
                kid[:, dirtybit1] = people[p1:p1 + 1, :, dirtybit1]
                kid[:, dirtybit2] = people[p2:p2 + 1, :, dirtybit2]
                dirtmap_kids.append(list(set(dirtybit2 + dirtybit1)))
                kids.append(kid)

            kids = np.stack(kids)
            kids, dirtmap_kids = self._mutation(kids, dirtmap_kids, self._mutation_rate)

            # For each child, check whether too many pixels have been changed
            for i in range(kids.shape[0]):
                if len(dirtmap_kids[i]) > self._max_pixels:
                    fix_pixel = np.random.choice(dirtmap_kids[i], len(dirtmap_kids[i]) - self._max_pixels, replace=False)
                    kids[i:i + 1, :, fix_pixel] = adv_img[:, fix_pixel]
                    dirtmap_kids[i] = [x for x in dirtmap_kids[i] if x not in fix_pixel]

            people = np.concatenate((kids, people[elite:elite + 1, :]))
            dirtmap_kids.append(dirtmap[elite])
            dirtmap = dirtmap_kids

        return people, dirtmap


