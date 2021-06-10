from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FrequencyModulatedKernelBO.experiments.nas.nasnet_ops import NASNET_OPS
from FrequencyModulatedKernelBO.experiments.nas.nasnet_block import NASNetBlock
from FrequencyModulatedKernelBO.experiments.nas.utils import initialize_conv2d, initialize_batchnorm2d, initialize_linear


class NASNet(nn.Module):
    def __init__(self,
                 block_architecture: List[Tuple[Tuple[int, str], Tuple[int, str]]], block_output_state: int,
                 input_dim: List[int], n_channel_list: List[int], num_classes: int):
        super().__init__()
        assert len(input_dim) == 3
        n_input_channel, input_h, input_w = input_dim
        assert input_h == input_w
        assert input_h in [28, 32]
        pool_padding = 0 if input_h == 32 else 1
        self._n_channel_list = n_channel_list
        setattr(self, 'cell%02d' % 0, nn.Conv2d(in_channels=n_input_channel, out_channels=16,
                                                kernel_size=3, padding=1, bias=True))
        in_channels = 16
        for i, n_channels in enumerate(self._n_channel_list):
            out_channels = n_channels
            if i > 0:
                setattr(self, 'cell%02d-pre-pool1' % (i + 1),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=pool_padding))
                setattr(self, 'cell%02d-pre-pool2' % (i + 1),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=pool_padding))
                input_h, input_w = input_h // 2 + pool_padding, input_w // 2 + pool_padding
            setattr(self, 'cell%02d-pre-conv1' % (i + 1), nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                                   kernel_size=1, bias=True))
            setattr(self, 'cell%02d-pre-conv2' % (i + 1), nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                                    kernel_size=1, bias=True))
            setattr(self, 'cell%2d' % (i + 1), NASNetBlock(block_architecture=block_architecture,
                                                           block_output_state=block_output_state,
                                                           n_channel=out_channels))
            in_channels = out_channels
        setattr(self, 'last_pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        setattr(self, 'last_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        input_h, input_w = input_h // 2, input_w // 2
        setattr(self, 'last_conv', nn.Conv2d(in_channels=n_channel_list[-1] * 2, out_channels=n_channel_list[-1],
                                             kernel_size=1, bias=True))
        setattr(self, 'last_bn', nn.BatchNorm2d(num_features=n_channel_list[-1]))
        setattr(self, 'last_fc', nn.Linear(in_features=n_channel_list[-1] * int(input_h * input_w),
                                           out_features=num_classes, bias=True))

    def forward(self, x):
        x1 = getattr(self, 'cell%02d' % 0)(x)
        x2 = x1
        for i, n_channels in enumerate(self._n_channel_list):
            if i > 0:
                x1 = getattr(self, 'cell%02d-pre-pool1' % (i + 1))(x1)
                x2 = getattr(self, 'cell%02d-pre-pool2' % (i + 1))(x2)
            x1 = getattr(self, 'cell%02d-pre-conv1' % (i + 1))(x1)
            x2 = getattr(self, 'cell%02d-pre-conv2' % (i + 1))(x2)
            x1, x2 = getattr(self, 'cell%2d' % (i + 1))(x1, x2)
        x1 = getattr(self, 'last_pool1')(x1)
        x2 = getattr(self, 'last_pool2')(x2)
        x = getattr(self, 'last_bn')(F.relu(getattr(self, 'last_conv')(torch.cat([x1, x2], dim=1)), inplace=True))
        x = getattr(self, 'last_fc')(x.view(x.size(0), -1))
        return x

    def initialize_params(self):
        initialize_conv2d(getattr(self, 'cell%02d' % 0))
        for i, n_channels in enumerate(self._n_channel_list):
            initialize_conv2d(getattr(self, 'cell%02d-pre-conv1' % (i + 1)))
            initialize_conv2d(getattr(self, 'cell%02d-pre-conv2' % (i + 1)))
            getattr(self, 'cell%2d' % (i + 1)).initialize_params()
        initialize_conv2d(getattr(self, 'last_conv'))
        initialize_batchnorm2d(getattr(self, 'last_bn'))
        initialize_linear(getattr(self, 'last_fc'))
