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

"""Attack by applying a patch on the image."""

import warnings
import logging
import numpy as np
from tqdm import tqdm
from .base import Metric
from .base import call_decorator
from perceptron.utils.image import onehot_like
from perceptron.utils.func import to_tanh_space
from perceptron.utils.func import to_model_space
from perceptron.utils.func import AdamOptimizer


class PatchVanishMetric(Metric):
    """Applying patch aiming at making object disappear."""

    @call_decorator
    def __call__(self, adv, mask=None, annotation=None, unpack=True,
                 binary_search_steps=5, max_iterations=1000,
                 confidence=0, learning_rate=5e-3,
                 initial_const=1e-2, abort_early=True):
        """Patch that vanishes object using Carlini & Wagner to increase loss.

        Parameters
        ----------
        adv : :class:`Adversarial`
            An :class:`Adversarial` instance.
        mask : tuple or list
            The region that will apply attacks [top, left, bottom, right].
        annotation : int
            The reference label of the original input.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search_steps : int
            The number of steps for the binary search used to find the
            optimal tradeoff-constant between distance and confidence.
        max_iterations : int
            The maxinum number of iterations. Largert values are more
            accurate; setting it too small will require a large learning
            rate and will produce poor results.
        confidence : int or float
            Confidence of adversarial examples: a higher value produces
            adversarials that are further away, but more strongly classified
            as adversarial.
        learning_rate : float
            The learning rate for the attack algorithm. Smaller values
            produce better results but take longer to converge.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative
            importance of distance and confidenc. If `binary_search_steps`
            is large, the initial constant is not important.
        abort_early : bool
            If True, Adam will be aborted if the loss hasn't decreased for
            some time (a tenth of max_iterations).
        """

        a = adv

        del adv
        del annotation
        del unpack

        image = a.original_image
        axis = a.channel_axis(batch=False)
        hw = [image.shape[i] for i in range(image.ndim) if i != axis]
        img_height, img_width = hw

        if mask is None:
            mask = np.array([0, 0, img_height - 1, img_width - 1])
        mask = np.array(mask).astype(int)

        mask_img = np.zeros((a.original_image.shape))
        if axis == 0:
            mask_img[:, mask[0]: mask[2] + 1, mask[1]: mask[3] + 1] = 1
        elif axis == 2:
            mask_img[mask[0]: mask[2] + 1, mask[1]: mask[3] + 1, :] = 1

        if not a.has_gradient():
            logging.fatal('Applied gradient-based attack to model that '
                          'does not provide gradients.')
            return

        min_, max_ = a.bounds()

        if a.model_task() == 'cls':
            loss_and_gradient = self.cls_loss_and_gradient
        elif a.model_task() == 'det':
            loss_and_gradient = self.det_loss_and_gradient
        else:
            raise ValueError('Model task not supported. Check that the'
                             ' task is either cls or det')
        # variables representing inputs in attack space will be
        # prefixed with att_

        att_original = to_tanh_space(a.original_image, min_, max_)

        # will be close but not identical to a.original_image
        reconstructed_original, _ = to_model_space(att_original, min_, max_)

        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_const
        lower_bound = 0
        upper_bound = np.inf

        for binary_search_step in tqdm(range(binary_search_steps)):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                const = upper_bound

            logging.info('starting optimization with const = {}'.format(const))
            att_perturbation = np.zeros_like(att_original)

            # create a new optimizer to minimize the perturbation
            optimizer = AdamOptimizer(att_perturbation.shape)

            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf

            for iteration in range(max_iterations):
                x, dxdp = to_model_space(
                    att_original + att_perturbation, min_, max_)

                loss, gradient, is_adv = loss_and_gradient(
                    const, a, x, dxdp, reconstructed_original,
                    confidence, min_, max_)

                for idx_0 in range(mask_img.shape[0]):
                    for idx_1 in range(mask_img.shape[1]):
                        for idx_2 in range(mask_img.shape[2]):
                            if mask_img[idx_0, idx_1, idx_2] == 0:
                                gradient[idx_0, idx_1, idx_2] = 0

                logging.info('iter: {}; loss: {}; best overall distance: {}'.format(
                    iteration, loss, a.distance))

                att_perturbation += optimizer(gradient, learning_rate)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if abort_early and \
                        iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if not (loss <= .9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

            if found_adv:
                logging.info('found adversarial with const = {}'.format(const))
                upper_bound = const
            else:
                logging.info('failed to find adversarial '
                             'with const = {}'.format(const))
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

    @classmethod
    def lp_distance_and_grad(cls, reference, other, span):
        """Calculate Linf distance and gradient."""
        diff = np.abs((other - reference))
        max_diff = np.max(diff)
        l_inf_distance = max_diff / span
        if (max_diff == 0):
            l_inf_distance_grad = np.zeros_like(diff, dtype=np.float32)
        else:
            l_inf_distance_grad = (diff == max_diff).astype(np.float32)
        return l_inf_distance, l_inf_distance_grad

    @classmethod
    def det_loss_and_gradient(cls, const, a, x, dxdp,
                              reconstructed_original, confidence, min_, max_):
        """
        Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x).
        """

        _, is_adv_loss, is_adv_loss_grad, is_adv = \
            a.predictions_and_gradient(x)

        targeted = a.target_class() is not None
        if targeted:
            c_minimize = a.target_class()
        else:
            raise NotImplementedError

        # is_adv_loss, is_adv_loss_grad = a.backward(c_minimize, x)

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence

        is_adv_loss += confidence
        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        squared_lp_distance, squared_lp_distance_grad = \
            cls.lp_distance_and_grad(reconstructed_original, x, s)

        total_loss = squared_lp_distance + const * is_adv_loss
        total_loss_grad = squared_lp_distance_grad + const * is_adv_loss_grad

        # backprop the gradient of the loss w.r.t. x further
        # to get the gradient of the loss w.r.t. att_perturbation
        assert total_loss_grad.shape == x.shape
        assert dxdp.shape == x.shape
        # we can do a simple elementwise multiplication, because
        # grad_x_wrt_p is a matrix of elementwise derivatives
        # (i.e. each x[i] w.r.t. p[i] only, for all i) and
        # grad_loss_wrt_x is a real gradient reshaped as a matrix
        gradient = total_loss_grad * dxdp

        return total_loss, gradient, is_adv

    @classmethod
    def cls_loss_and_gradient(cls, const, a, x, dxdp,
                              reconstructed_original, confidence, min_, max_):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x).
        """

        logits, is_adv = a.predictions(x)

        targeted = a.target_class() is not None
        if targeted:
            c_minimize = cls.best_other_class(logits, a.target_class())
            c_maximize = a.target_class()
        else:
            c_minimize = a.original_pred
            c_maximize = cls.best_other_class(logits, a.original_pred)

        is_adv_loss = logits[c_minimize] - logits[c_maximize]

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence

        is_adv_loss += confidence
        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        lp_distance, lp_distance_grad = \
            cls.lp_distance_and_grad(reconstructed_original, x, s)

        total_loss = lp_distance + const * is_adv_loss

        # calculate the gradient of total_loss w.r.t. x
        logits_diff_grad = np.zeros_like(logits)
        logits_diff_grad[c_minimize] = 1
        logits_diff_grad[c_maximize] = -1
        is_adv_loss_grad = a.backward(logits_diff_grad, x)
        assert is_adv_loss >= 0
        if is_adv_loss == 0:
            is_adv_loss_grad = 0

        total_loss_grad = lp_distance_grad + const * is_adv_loss_grad
        # backprop the gradient of the loss w.r.t. x further
        # to get the gradient of the loss w.r.t. att_perturbation
        assert total_loss_grad.shape == x.shape
        assert dxdp.shape == x.shape
        # we can do a simple elementwise multiplication, because
        # grad_x_wrt_p is a matrix of elementwise derivatives
        # (i.e. each x[i] w.r.t. p[i] only, for all i) and
        # grad_loss_wrt_x is a real gradient reshaped as a matrix
        gradient = total_loss_grad * dxdp

        return total_loss, gradient, is_adv

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`.
        """
        other_logits = logits - onehot_like(logits, exclude, value=np.inf)
        return np.argmax(other_logits)
