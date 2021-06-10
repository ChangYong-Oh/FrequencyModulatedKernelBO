#!/usr/bin/env python3

import torch
from torch.nn import Module as TModule
from torch.distributions import Beta

from gpytorch.priors.prior import Prior
from gpytorch.priors.utils import _bufferize_attributes


class BetaPrior(Prior, Beta):
    """
    """

    def __init__(self, concentration1, concentration0, validate_args=False, transform=None):
        TModule.__init__(self)
        Beta.__init__(self, concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return BetaPrior(self.concentration1.expand(batch_shape), self.concentration0.expand(batch_shape))
