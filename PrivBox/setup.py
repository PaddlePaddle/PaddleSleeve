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
Setup file for python code install
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

from setuptools import find_packages
from setuptools import setup
from version import priv_box_version

def python_version():
    """
    get python version
    """
    return [int(v) for v in platform.python_version().split(".")]


max_version, mid_version, min_version = python_version()

REQUIRED_PACKAGES = [
    'six >= 1.10.0', 'protobuf >= 3.1.0', 'paddlepaddle >= 2.0.0', 'paddlepaddle-gpu >= 2.0.0'
]

if max_version < 3:
    REQUIRED_PACKAGES += ["enum"]
else:
    REQUIRED_PACKAGES += ["numpy"]

REQUIRED_PACKAGES += ["unittest2"]
packages = ['privbox', 'privbox.inference', 'privbox.inference.membership_inference', 'privbox.inference.property_inference',
            'privbox.extraction', 'privbox.inversion', 'privbox.dataset', 'privbox.metrics']
package_data = {}
package_dir = {
    'privbox': '.',
    'privbox.inference': './inference',
    'privbox.inference.membership_inference': './inference/membership_inference',
    'privbox.inference.property_inference': './inference/property_inference',
    'privbox.extraction': './extraction',
    'privbox.inversion': './inversion',
    'privbox.dataset': './dataset',
    'privbox.metrics': './metrics'
}


setup(
    name='privbox',
    version=priv_box_version.replace('-', ''),
    description=(
        'a Python library based on PaddlePaddle for testing AI model privacy leaking risk.'),
    long_description='',
    url='https://github.com/PaddlePaddle/PaddleSleeve',
    author='PaddlePaddle Author',
    author_email='paddle-dev@baidu.com',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    package_data=package_data,
    package_dir=package_dir,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=(
        'paddlesleeve privacy-box membership-inference extraction inversion AI-attack paddlepaddle'
    ))
