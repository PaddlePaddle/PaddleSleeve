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
unittest for AT&T face dataset
"""
from __future__ import print_function
from six.moves import range
from PIL import Image, ImageOps

import unittest
import gzip
import numpy as np
import argparse
import struct
import os
import paddle
import random
from privbox.dataset import ATTFace

import paddle.vision.transforms as T

class TestATTFaceTrain(unittest.TestCase):
    """
    ATTFace test
    """
    def test_main(self):
        """
        main test
        """
        #transform = T.Transpose([0, 2, 1])
        att_face = ATTFace(mode="train", transform=None)
        self.assertTrue(len(att_face) == 400)
        for i in range(len(att_face)):
            img, lab = att_face[i]
            self.assertTrue(img.shape[0] == 112)
            self.assertTrue(img.shape[1] == 92)
            self.assertTrue(lab.shape[0] == 1)
            self.assertTrue(0 <= int(lab) <= 40)


if __name__ == '__main__':
    unittest.main()
