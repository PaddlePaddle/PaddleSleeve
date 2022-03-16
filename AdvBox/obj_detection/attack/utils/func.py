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

import os
import numpy as np


def softmax(logits):
    """Transforms predictions into probability values."""
    assert logits.ndim == 1

    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def crossentropy(label, logits):
    """Calculates the cross-entropy.
    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.
    label : int
        The label describing the target distribution.
    Returns
    -------
    float: The cross-entropy between softmax(logits) and onehot(label).
    """

    assert logits.ndim == 1

    logits = logits - np.max(logits)
    e = np.exp(logits)
    s = np.sum(e)
    ce = np.log(s) - logits[label]
    return ce


def batch_crossentropy(label, logits):
    """Calculates the cross-entropy for a batch of logits.
    Parameters
    ----------
    logits : array_like
        The logits predicted by the model for a batch of inputs
    label : int
        The label describing the target distribution.
    Returns
    -------
    np.ndarray
        The cross-entropy between softmax(logits[i]) and onehot(label)
        for all i.
    """

    assert logits.ndim == 2

    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits)
    s = np.sum(e, axis=1)
    ces = np.log(s) - logits[:, label]
    return ces


def binarize(x, values, threshold=None, included_in='upper'):
    """Binarizes the value of x.
    Parameters
    ----------
    values : tuple of two floats
        The lower and upper value to which the inputs are mapped.
    threshold : float
        The threshold;l defaults to (values[0] + values[1]) / 2 if None.
    included_in : str
        Whether the threshold value itself belongs to the lower or
        upper interval.
    """
    lower, upper = values

    if threshold is None:
        threshold = (lower + upper) / 2.

    x = x.copy()
    if included_in == 'lower':
        x[x <= threshold] = lower
        x[x > threshold] = upper
    elif included_in == 'upper':
        x[x < threshold] = lower
        x[x >= threshold] = upper
    else:
        raise ValueError('included_in must be "lower" or "upper"')
    return x


def to_tanh_space(x, min_, max_):
    """Convert an input from model space to tanh space."""
    # map from [min_, max_] to [-1, +1]
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b

    # from [-1, +1] to approx. (-1, +1)
    x = x * 0.999999

    # from (-1, +1) to (-inf, +inf)
    return np.arctanh(x)


def to_model_space(x, min_, max_):
    """Convert an input from tanh space to model space."""
    x = np.tanh(x)

    grad = 1 - np.square(x)

    # map from (-1, +1) to (min_, max_)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a

    grad = grad * b
    return x, grad


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized.
    """

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def __call__(self, gradient, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=10e-8):
        """Updates internal parameters of the optimizer and returns the
        change that should be applied to the variable.
        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loww w.r.t. to the variable.
        learning_rate: float
            the learning rate in the current iteration.
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients.
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients.
        epsilon: float
            small value to avoid division by zero.
        """
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


def maybe_download_model_data(file_name, url_string):
    import sys
    import tempfile
    try:
        from urllib.parse import urljoin
        from urllib.request import urlretrieve
    except ImportError:
        from urlparse import urljoin
        from urllib import urlretrieve
    data_dir = tempfile.gettempdir()
    dest_file = os.path.join(data_dir, file_name)
    isfile = os.path.isfile(dest_file)
    if not isfile:
        url = urljoin(
            url_string,
            file_name)
        print('Downloading %s' % file_name)

        def dlProgress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\r" + url + "...%d%%" % percent)
            sys.stdout.flush()
        urlretrieve(url, dest_file, reporthook=dlProgress)
    return dest_file


def maybe_download_image(file_name, url_string):
    import sys
    try:
        from urllib.parse import urljoin
        from urllib.request import urlretrieve
    except ImportError:
        from urlparse import urljoin
        from urllib import urlretrieve
    dest_file = path = os.path.join(os.path.dirname(__file__), 'images/%s' % file_name)
    isfile = os.path.isfile(dest_file)
    if not isfile:
        url = urljoin(
            url_string,
            file_name)
        print('Downloading %s' % file_name)

        def dlProgress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\r" + url + "...%d%%" % percent)
            sys.stdout.flush()
        urlretrieve(url, dest_file, reporthook=dlProgress)
    return dest_file


def bound(image, model, sd, num_class, num_iter=10):
    from scipy.stats import norm
    one_hot = np.zeros(num_class)
    for i in range(num_iter):
        noise = np.random.normal(scale=sd, size=image.shape).astype(np.float32)
        logits = model.predictions(image+noise)
        one_hot[logits.argmax()] += 1
    ret = sorted(one_hot/np.sum(one_hot))[::-1]
    qi = ret[0]-1e-9
    qj = ret[1]+1e-9
    return sd/2.*(norm.ppf(qi)-norm.ppf(qj))

