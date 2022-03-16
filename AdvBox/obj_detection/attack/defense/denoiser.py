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
paddle2 model high-level representation guided denoiser(HGD) demo.
* implemented Denoising U-NET(DUNET)
"""

import paddle
import paddle.nn as nn
from ppdet.modeling.backbones.darknet import ConvBNLayer
import sys
sys.path.append("../..")

USE_GPU = paddle.get_device()
if USE_GPU.startswith('gpu'):
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
paddle.seed(2021)


class C2Block(nn.Layer):

    def __init__(self,
                 ch_in,
                 ch_out,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        C2 Block layer of DUNET, consisted of 2 ConvBN layers
        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """

        super(C2Block, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)

    def forward(self, inputs):
        outs = self.conv1(inputs)
        outs = self.conv2(outs)
        return outs


class C3Block(nn.Layer):

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride=2,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        C3 Block layer of DUNET, consisted of 3 ConvBN layers
        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """

        super(C3Block, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            padding=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.c2 = C2Block(
            ch_in=ch_out,
            ch_out=ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)

    def forward(self, inputs):
        outs = self.conv1(inputs)
        outs = self.c2(outs)
        return outs


class FuseConv(nn.Layer):

    def __init__(self,
                 ch_in,
                 ch_out,
                 scale=2,
                 num_conv=3,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        Fuse and Conv Block of DUNET
        Args:
            ch_in (list): input channel of feedback path input and lateral input
            ch_out (int): output channels of fuse and conv layers
            scale(int): scale factor of upsampling
            num_conv(int): number of conv layers after fuse unit
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """

        super(FuseConv, self).__init__()

        self.bi = paddle.nn.UpsamplingBilinear2D(scale_factor=scale, data_format=data_format)
        conv_in = int(sum(ch_in))
        if num_conv == 3:
            self.conv = C3Block(ch_in=conv_in,
                                ch_out=int(ch_out),
                                stride=1,
                                norm_type=norm_type,
                                norm_decay=norm_decay,
                                freeze_norm=freeze_norm,
                                data_format=data_format)
        else:
            self.conv = C2Block(ch_in=conv_in,
                                ch_out=int(ch_out),
                                norm_type=norm_type,
                                norm_decay=norm_decay,
                                freeze_norm=freeze_norm,
                                data_format=data_format)

    def forward(self, input1, input2):
        input1 = self.bi(input1)
        input1 = paddle.concat([input1, input2], axis=1)
        input1 = self.conv(input1)
        return input1


class DUNET(nn.Layer):

    def __init__(self,
                 num_blocks=4,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        Denoising U-NET Model.
        Reference Implementation: https://arxiv.org/pdf/1712.02976
        Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser

        Author: Liao Fangzhou.
        Args:
            ch_in (int): input channel
            ch_out (list): output channels of two ConvBN layers
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """

        super(DUNET, self).__init__()
        self.num_blocks = num_blocks
        ch_in = [64, 128, 256, 256]
        ch_out = [128, 256, 256, 256]

        self.conv0 = C2Block(ch_in=3,
                             ch_out=ch_in[0],
                             norm_type=norm_type,
                             norm_decay=norm_decay,
                             freeze_norm=freeze_norm,
                             data_format=data_format)
        self._forward_list = []
        self._backward_list = []
        for i in range(self.num_blocks):
            name = 'forward.{}.conv'.format(i)
            conv_block = self.add_sublayer(
                name,
                C3Block(
                    ch_in=int(ch_in[i]),
                    ch_out=int(ch_out[i]),
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    data_format=data_format))
            self._forward_list.append(conv_block)
        for i in range(self.num_blocks):
            name = 'backward.{}.fuse'.format(i)
            num_conv = 3 if i < self.num_blocks - 1 else 2
            fuse_conv = self.add_sublayer(
                name,
                FuseConv(
                    ch_in=[int(ch_out[-i - 1]), int(ch_in[-i - 1])],
                    ch_out=int(ch_in[-i - 1]),
                    num_conv=num_conv,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    data_format=data_format))
            self._backward_list.append(fuse_conv)

        self.out_conv = nn.Conv2D(in_channels=ch_in[0],
                                  out_channels=3,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  data_format=data_format,
                                  bias_attr=False)

    def forward(self, inputs):
        """

        Parameters
        ----------
        inputs (Dict) : including :
                                    image (Tensor) : input image in NCHW format
                                    im_shape (Tensor) : shape of the image
                                    scale_factor (Tensor) : the scale factor
        Returns
        -------
        denoised (Tensor) : the image after denoising in NCHW format

        """
        x = inputs['image']
        out = self.conv0(x)

        lateral_inputs = []
        for i, forward_block in enumerate(self._forward_list):
            lateral_inputs.append(out)
            out = forward_block(out)

        for i, backward_block in enumerate(self._backward_list):
            lateral_input = lateral_inputs[-i-1]
            out = backward_block(out, lateral_input)

        out = self.out_conv(out)
        inputs['image'] = paddle.add(x, out * 0.1)

        return inputs
