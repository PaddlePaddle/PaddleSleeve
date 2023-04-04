# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base logic for input depreprocess.

Author: tianweijuan
"""
import traceback

from ppdet.data.transform import operators

# TODO: set logger
# from ..ppdet.utils.logger import setup_logger
# logger = setup_logger('reader')

class OperatorComposer(object):
    def __init__(self, operators, num_classes=80):
        self.operators = operators
        self.operators_cls = []
        for operator_name in self.operators:
            op_cls = getattr(operators, operator_name)
            f = op_cls(**self.operators[operator_name])
            if hasattr(f, 'num_classes'):
                f.num_classes = num_classes

            self.operators_cls.append(f)

    def __call__(self, data):
        for f in self.operators_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                # logger.warning("fail to map sample transform [{}] "
                #                "with error: {} and stack:\n{}".format(
                #                    f, e, str(stack_info)))
                raise e

        return data


