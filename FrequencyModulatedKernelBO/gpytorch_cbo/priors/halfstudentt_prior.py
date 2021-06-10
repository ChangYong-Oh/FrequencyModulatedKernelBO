#!/usr/bin/env python3

import math

import torch
from torch.distributions import StudentT, constraints
from torch.nn import Module as TModule

from gpytorch.priors.prior import Prior


class HalfStudentTPrior(Prior, StudentT):
    arg_constraints = {'df': constraints.positive, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, df, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        StudentT.__init__(self, df=df, loc=0, scale=scale, validate_args=validate_args)
        self._transform = transform

    def rsample(self, sample_shape=torch.Size([])):
        return torch.abs(StudentT.rsample(sample_shape=sample_shape))

    def expand(self, expand_shape, _instance=None):
        new = self._get_checked_instance(HalfStudentTPrior, _instance)
        batch_shape = torch.Size(expand_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(HalfStudentTPrior, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, X):
        return StudentT.log_prob(X) + math.log(2)
