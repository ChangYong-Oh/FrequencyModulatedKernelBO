#!/usr/bin/env python3

import torch
from torch.nn import Module as TModule
from pyro.distributions import InverseGamma

from gpytorch.priors.prior import Prior
from gpytorch.priors.utils import _bufferize_attributes


class InverseGammaPrior(Prior, InverseGamma):
    """InverseGamma Prior parameterized by concentration and rate

    pdf(x) = beta^alpha / Gamma(alpha) * x^(-alpha - 1) * exp(-beta / x)

    were alpha(concentration) > 0 and beta(rate) > 0 are the concentration and rate parameters, respectively.
    """

    def __init__(self, concentration, rate, validate_args=False, transform=None):
        TModule.__init__(self)
        InverseGamma.__init__(self, concentration=concentration, rate=rate, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return InverseGammaPrior(self.concentration.expand(batch_shape), self.rate.expand(batch_shape))
