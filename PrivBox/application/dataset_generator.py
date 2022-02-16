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

import sys
import re
import paddle
import yaml
import os
import logging
import importlib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetGenerator(object):
    """R
    generate dataset object
    """
    def gen(self, dataset_dir):
        """R
        generate dataset of type paddle.io.dataset
        """
        files = os.listdir(dataset_dir)
        dataset_file = None
        for name in files:
            if re.match(".*\.py$", name):
                if dataset_file is not None:
                    raise RuntimeError("More than one paddle dataset file (.py file) in dataset dir")
                dataset_file = name

        if dataset_file is None:
            raise RuntimeError("Must have paddle dataset file in dataset dir: " + dataset_dir)
        
        sys.path.append(dataset_dir)
        dataset = self._get_dataset(dataset_file[:-3])
        
        logger.info("Init dataset finish.")
        return [dataset, dataset_file[:-3]]

    def _get_dataset(self, dataset_module_name):
        logger.info("Init dataset " + dataset_module_name)

        dataset_module = importlib.import_module(dataset_module_name)
        
        dataset = getattr(dataset_module, "get_dataset")()
        if dataset is None:
            raise RuntimeError(
                "`get_dataset()` method must be defined in dataset file: " + dataset_module_name + ".py")

        return dataset


