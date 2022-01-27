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

class ModelGenerator(object):
    """R
    generate model object
    """
    def gen(self, model_dir, pre_train=True):
        """R
        generate network of type paddle.nn.Layer
        """
        files = os.listdir(model_dir)
        net_file = None
        layer_dict_file = None
        for name in files:
            if re.match(".*\.py$", name):
                if net_file is not None:
                    raise RuntimeError("More than one paddle net file (.py file) in model dir")
                net_file = name

            elif re.match(".*\.pdparams", name):
                if layer_dict_file is not None:
                    raise RuntimeError("More than one model parameter file (.pdparams file) in model dir")
                layer_dict_file = name

        if net_file is None:
            raise RuntimeError("Must have paddle net file in model dir: " + model_dir)
        
        sys.path.append(model_dir)
        
        net = self._get_net(net_file[:-3])
        if layer_dict_file and pre_train:
            logger.info("Load layer dict from file: " + model_dir + "/" + layer_dict_file)
            layer_state_dict = paddle.load(model_dir + "/" + layer_dict_file)
            net.set_state_dict(layer_state_dict)
        logger.info("Init model done.")
        return [net, net_file[:-3]]

    def _get_net(self, net_module_name):

        logger.info("Init model " + net_module_name)

        net_module = importlib.import_module(net_module_name)

        net = getattr(net_module, "get_model")()

        if net is None:
            raise RuntimeError(
                "`get_model()` method must be defined in model file: " + net_module_name + ".py")

        return net


