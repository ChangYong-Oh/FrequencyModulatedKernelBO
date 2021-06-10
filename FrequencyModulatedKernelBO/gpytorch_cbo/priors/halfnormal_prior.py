#!/usr/bin/env python3

import math

import torch
from torch.distributions import HalfNormal, constraints
from torch.nn import Module as TModule

from gpytorch.priors.prior import Prior


class HalfNormalPrior(Prior, HalfNormal):
    def __init__(self, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        HalfNormal.__init__(self, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        return HalfNormalPrior(self.scale.expand(batch_shape))
