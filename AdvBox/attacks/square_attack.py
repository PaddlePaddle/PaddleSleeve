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
This module provides the implementation for square attack method.

"Square Attack Method" was originally implemented by Andriushchenko et
    al. (2020)

    Paper link: https://arxiv.org/abs/1912.00049

"""
from __future__ import division
import numpy as np
import paddle
from .base import Attack


__all__ = [
    'SquareAttack2', 'SQ_2',
    'SquareAttackInfinity', 'SQ_Inf'
]


class SquareAttack2(Attack):
    """
    This class implements square attack method when L2-Norm is used
    """
    def __init__(self, model, target=-1, max_steps=1000, eps=0.05, eps_step=0, threshold=-1, window_size=0.1):
        """
        Args:
            model: An instance of a paddle model to be attacked.
            max_steps(int): Max iteration steps allowed
            eps(float): Maximum amplitude of perturbation
            window_size(float): size of the perturbation window, represented as a fraction of the original image size
        """
        super(SquareAttack2, self).__init__(model)
        self._max_steps = max_steps
        self._window_size = window_size
        self.lb, self.ub = self.model.bounds
        self._eps = eps * (self.ub - self.lb)
        self._eps_step = eps_step
        self._threshold = threshold if threshold > 0 else self._max_steps
        self.target = target
        self.best = None
        self.best_perf = -np.inf
        self.success_noise = None

    def _adjust_size(self, step):
        assert 0 <= step <= 1, 'Invalid current step number'

        if step < 0.05:
            return self._window_size
        elif step < 0.2:
            return self._window_size / 2
        elif step < 0.5:
            return self._window_size / 4
        elif step < 0.8:
            return self._window_size / 8
        else:
            return self._window_size / 16


    def _pseudo_guassian(self, x, y):
        """
        Args:
            x(int): width of the noise window
            y(int): height of the noise window
        """
        n = x // 2
        res = np.zeros((y, x))

        x1, x2 = x // 2, x // 2 + 1
        y1, y2 = y // 2 , y // 2 + 1
        for k in np.arange(n, -1, -1):
            inc = 1 / (n+1-k) ** 2
            res[y1:y2, x1:x2] += inc
            #update window bound
            x1, x2 = np.clip([x1-1, x2+1], 0, x)
            y1, y2 = np.clip([y1-1, y2+1], 0, y)

        return res / np.linalg.norm(res)




    def _generate_noise(self, h):
        """
        Args:
            h(int): size of the perturbation window, represented as the number of pixels
        """
        eta = np.zeros((h, h))
        n = h // 2

        eta[:n, :] = self._pseudo_guassian(h, n)
        eta[n:, :] = -1 * self._pseudo_guassian(h, h-n)

        eta = np.transpose(eta) if np.random.choice([-1,1], 1) > 0 else eta
        return eta / np.linalg.norm(eta)



    def _apply(self, adversary):
        """
        Apply the square attack method with L2

        Args:
            adversary: The Adversary object.
        Returns:
            adversary(Adversary): The Adversary object.
        """



        if adversary.is_targeted_attack:
            self.target = adversary.target_label
            num_labels = self.model.num_classes()
            assert self.target < num_labels

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.denormalized_original)
        assert adv_img.ndim == 3, 'Invalid adversary, the original image should be in CHW format'

        c, h, w = adv_img.shape

        # Add initial noise
        noise = np.zeros(adv_img.shape)
        s = h // 4  # The initial window size
        r0 = (h - 4 * s) // 2
        for i in range(h // s):
            x1 = s * i + r0
            x2 = x1 + s
            for j in range(w // s):
                y1 = s * j + r0
                y2 = y1 + s
                noise[:,x1:x2, y1:y2] = np.random.choice([-1,1], size=(c, 1, 1), replace=True) * \
                                        self._generate_noise(s).reshape((1, s, s))
        cur_noise = noise / np.linalg.norm(noise) * self._eps
        self.best_perf = -np.inf
        eps_avail = 0
        p_step = 0
        remain_steps = self._max_steps
        plateau_time = 0

        # The Main Loop
        for step in range(self._max_steps):
            if plateau_time > self._threshold:
                self._eps += self._eps_step 
                plateau_time = 0
                p_step = 0
                print('Warning: Attack Failed, now search in eps={0}'.format(self._eps))

            # Add normalized noise to the original image
            img = adv_img + cur_noise / np.linalg.norm(cur_noise) * self._eps
            img = np.clip(img, self.lb, self.ub)
            x = paddle.to_tensor(img, dtype='float32', place=self._device)
            x = self.input_preprocess(x)
            label = self.model.predict(x)[0, :]
            tag = np.argmax(label)

            if self.target == -1:
                if tag != adversary.original_label:
                    norm = self.safe_delete_batchsize_dimension(x).numpy()
                    is_ok = adversary.try_accept_the_example(img, norm, tag)

                    if is_ok:
                        self.success_noise = cur_noise
                        return adversary
            else:
                if tag == self.target:
                    norm = self.safe_delete_batchsize_dimension(x).numpy()
                    is_ok = adversary.try_accept_the_example(img, norm, tag)

                    if is_ok:
                        self.success_noise = cur_noise
                        return adversary


            # Calculate loss to see if there is any progress
            if self.target != -1:
                score = label[self.target]
            else:
                label -= label[tag]
                label[tag] = -np.inf
                score = np.max(label)

            if score > self.best_perf:  #and np.linalg.norm(cur_noise) - self._eps < 1e-15:
                self.best = img + 0
                self.best_perf = score
                plateau_time = 0
            else: 
                plateau_time += 1
 
            cur_noise = self.best - adv_img
            if step % 500 == 0:
                print(step,score,self.best_perf,self._eps)
            # Generate new noise
            p = self._adjust_size(p_step / self._max_steps)
            s = max(5, int(h * p))
            r1, r2, c1, c2 = [0,0,0,0]
            while np.abs(r2 - r1) < s and np.abs(c2 - c1) < s:
                r1, r2 = np.random.choice(range(h-s), 2)
                c1, c2 = np.random.choice(range(w-s), 2)
            eta = np.random.choice([-1, 1], (c, 1, 1), replace=True) * \
                   self._generate_noise(s).reshape((1, s, s))
            eps_avail = max(0, self._eps ** 2 - np.linalg.norm(cur_noise) ** 2)

            for channel in range(c):
                norm1 = np.linalg.norm(cur_noise[channel, r1:r1+s, c1:c1+s])
                if norm1 > 0: 
                    pert = eta[channel,:,:] + cur_noise[channel, r1:r1+s, c1:c1+s] / norm1
                else:
                    norm1 = 0
                    pert = eta[channel,:,:]
                norm2 = np.linalg.norm(cur_noise[channel, r2:r2+s, c2:c2+s])
                eps_temp = np.sqrt(norm1 ** 2 + norm2 ** 2 + eps_avail / c)
                cur_noise[channel, r1:r1+s, c1:c1+s] = np.clip(pert / np.linalg.norm(pert) * eps_temp, self.lb, self.ub)
                cur_noise[channel, r2:r2+s, c2:c2+s] = 0
            p_step += 1
            remain_steps -= 1

        self.success_noise = self.best - adv_img
        return adversary



class SquareAttackInfinity(Attack):
    """
    This class implements square attack method with infinity norm

    """
    def __init__(self, model, target=-1, max_steps=1000, eps=0.1, window_size=0.1):
        """
        Args:
            model: An instance of a paddle model to be attacked.
            max_steps(int): Max iteration steps allowed
            eps(float): Maximum amplitude of perturbation
            window_size(float): size of the perturbation window, represented as a fraction of the original image size
        """
        super(SquareAttackInfinity, self).__init__(model)
        self._max_steps = max_steps
        self._window_size = window_size
        self.lb, self.ub = self.model.bounds
        self.best = None
        self.best_perf = -np.inf
        self._eps = eps * (self.ub - self.lb)
        self.target = target
        self.success_noise = None

    def _adjust_size(self, step):
        assert 0 <= step <= 1, 'Invalid current step number'

        if step < 0.05:
            return self._window_size
        elif step < 0.1:
            return self._window_size / 2
        elif step < 0.2:
            return self._window_size / 4
        elif step < 0.5:
            return self._window_size / 8
        else:
            return self._window_size / 16

    def _apply(self, adversary):
        """
        Launch an attack process.
        Args:
            adversary: Adversary. An adversary instance with initial status.
            **kwargs: Other named arguments.

        Returns:
            An adversary status with changed status.
        """

        if adversary.is_targeted_attack:
            self.target = adversary.target_label
            num_labels = self.model.num_classes()
            assert self.target < num_labels

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.denormalized_original)
        assert adv_img.ndim == 3, 'Invalid adversary, the original image should be in CHW format'

        c, h, w = adv_img.shape

        # Add initial noise
        cur_noise = np.zeros(adv_img.shape)
        init_eta = np.random.choice([-self._eps, self._eps], size=[c, 1, w])
        init_eta = np.repeat(init_eta, h, axis=1)
        cur_noise = np.clip(cur_noise + init_eta, self.lb, self.ub)
        s = int(self._window_size * h)

        # Main Loop
        for step in range(self._max_steps):
            # test the current noise
            # Add normalized noise to the original image
            img = np.clip(adv_img + cur_noise, self.lb, self.ub)

            x = paddle.to_tensor(img, dtype='float32', place=self._device)
            x = self.input_preprocess(x)
            label = self.model.predict(x)[0, :]
            tag = np.argmax(label)

            if self.target == -1:
                if tag != adversary.original_label:
                    norm = self.safe_delete_batchsize_dimension(x).numpy()
                    is_ok = adversary.try_accept_the_example(img, norm, tag)

                    if is_ok:
                        self.success_noise = cur_noise
                        return adversary
            else:
                if tag == self.target:
                    norm = self.safe_delete_batchsize_dimension(x).numpy()
                    is_ok = adversary.try_accept_the_example(img, norm, tag)

                    if is_ok:
                        self.success_noise = cur_noise
                        return adversary

            # Calculate loss to see if there is any progress
            if self.target != -1:
                score = label[self.target]
            else:
                label -= label[tag]
                label[tag] = -np.inf
                score = np.max(label)

            if score > self.best_perf:
                self.best = img + 0
                self.best_perf = score
            cur_noise = self.best - adv_img
            if step % 500 == 0:
                print(step, self.best_perf)
            # Generate new noise
            p = self._adjust_size(step / self._max_steps)
            s = max(3, int(h * p))
            row = np.random.choice(range(h-s))
            col = np.random.choice(range(w-s))

            pert = np.random.choice([-self._eps, self._eps], (c,1,1), replace=True) * np.ones((1,s,s))
            cur_noise[:, row:row+s, col:col+s] = pert

        return adversary

SQ_2 = SquareAttack2
SQ_Inf = SquareAttackInfinity
