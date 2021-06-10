from typing import List, Optional

import pthflops

import torch
import torch.nn as nn


def initialize_conv2d(m: nn.Conv2d):
    if hasattr(m, 'bias'):
        torch.nn.init.zeros_(m.bias)
    torch.nn.init.kaiming_normal_(m.weight)

def initialize_batchnorm2d(m: nn.BatchNorm2d):
    torch.nn.init.uniform_(m.weight)
    torch.nn.init.zeros_(m.bias)
    torch.nn.init.zeros_(m.running_mean)
    torch.nn.init.ones_(m.running_var)

def initialize_linear(m: nn.Linear):
    if hasattr(m, 'bias'):
        torch.nn.init.zeros_(m.bias)
    torch.nn.init.kaiming_normal_(m.weight)

def compute_flops(m: nn.Module, input_dim: List):
    single_input = next(m.parameters()).new_zeros(size=[1] + input_dim)
    return pthflops.count_ops(model=m, input=single_input, print_readable=False, verbose=False)[0]


if __name__ == '__main__':
    m = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
    print('   %+.4E %+.4E %+.4E' %
          (torch.sum(m.weight).item(), torch.sum(m.weight ** 2).item(), torch.sum(m.weight ** 3).item()))
    for i in range(5):
        initialize_conv2d(m)
        print('%2d %+.4E %+.4E %+.4E' %
              (i, torch.sum(m.weight).item(), torch.sum(m.weight ** 2).item(), torch.sum(m.weight ** 3).item()))
