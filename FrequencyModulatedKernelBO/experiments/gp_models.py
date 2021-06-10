from copy import deepcopy

from typing import Dict, Optional, List

import sys
import time
import math
from pathos import multiprocessing
import multiprocess.context as ctx

import torch
from torch import Tensor

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel, AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import Prior, NormalPrior, LogNormalPrior
from gpytorch.constraints import Positive, Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

from FrequencyModulatedKernelBO.gpytorch_cbo.kernels import (
    ARDDiffusionKernel, ARDRegularizedLaplacianKernel,
    ModulatedDiffusionKernel, ModulatedRegularizedLaplacianKernel,
)
from FrequencyModulatedKernelBO.gpytorch_cbo.priors import \
    InverseGammaPrior, HalfHorseshoePrior, BetaPrior, HalfStudentTPrior, HalfNormalPrior
from pyro.infer.mcmc import NUTS, HMC
from FrequencyModulatedKernelBO.pyro_cbo.infer.mcmc import MCMC, Slice

ctx._force_start_method('spawn')

from torch.distributions import HalfNormal

# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================
# MOST OF PARAMETERS RELATED TO BAYESIAN OPTIMIZATION ARE COLLECTED BELOW, so only below can be modified for EXP
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ====================================================================================================================


# Surrogate model MCMC related design choices
MCMC_NUM_SAMPLES = 10
MCMC_WARMUP_STEPS = 5
OPTIMIZE_N_RUNS = 10
OPTIMIZE_N_CORES_USED = 16
OPTIMIZE_REL_TOL = 1e-4  # the larger this is, the smaller number of updates is performed until convergence
                         # this can trade off between efficiency and surrogate model fitting quality

# Surrogate model priors related design choices
# Among all kernel parameters, below parameters are set in advance and kept
# Others (constmean, noisevar, outputscale) given as arguments of GP class can be adjusted according to updated data


KERNEL_TYPE_LIST = ['Diffusion', 'Laplacian',
                    'ModulatedDiffusion', 'ModulatedLaplacian',
                    'AdditiveDiffusion', 'AdditiveLaplacian',
                    'ProductDiffusion', 'ProductLaplacian']
SAMPLER_TYPE_LIST = ['NUTS', 'SliceGroupwise', 'SliceElementwise']


def kernel_parameter_prior_alpha_diffusion(n_variables):
    return HalfHorseshoePrior(scale=0.1)


def kernel_parameter_prior_alpha_laplacian(n_variables):
    return HalfHorseshoePrior(scale=0.1)


def kernel_parameter_prior_beta_diffusion(n_variables):
    return HalfHorseshoePrior(scale=2.0)


def kernel_parameter_prior_beta_laplacian(n_variables):
    return HalfHorseshoePrior(scale=2.0)


def kernel_parameter_prior_lengthscale(n_continuous):
    return InverseGammaPrior(concentration=2.0, rate=0.5)


def kernel_parameter_prior_constmean(train_x: Optional[Tensor], train_y: Optional[Tensor]):
    assert (train_x is None) == (train_y is None)
    y_mean = 0 if train_x is None else torch.mean(train_y)
    y_std = 1 if train_x is None else torch.std(train_y)
    return NormalPrior(loc=y_mean, scale=y_std * 0.5)


def kernel_parameter_prior_noisevar(train_x: Tensor, train_y: Tensor):
    """
    Horseshoe causes a numerical instability, so avoiding zero prior is better while encouraging small value
    by choosing the rate small, InverseGammaPrior will not encourage a large value
    :param train_x:
    :param train_y:
    :return:
    """
    assert (train_x is None) == (train_y is None)
    train_y_var = 1.0 if train_x is None else torch.var(train_y).item()
    return InverseGammaPrior(concentration=2, rate=0.01 * train_y_var)


def kernel_parameter_prior_outputscale(model, sample: Dict, train_x: Optional[Tensor], train_y: Optional[Tensor]):
    """
    This heuristic assumes that train_y is normalized such that mean(train_y) = 0 and std(train_y)= 1
    :param train_x:
    :param train_y:
    :return:
    """
    assert (train_x is None) == (train_y is None)
    if train_x is None:
        return LogNormalPrior(loc=0, scale=3)
    assert isinstance(model, (CombinatorialGP, MixedVariableGP))
    if isinstance(model, (CombinatorialGP, ModulatedDiffusionGP, ProductDiffusionGP,
                          ModulatedLaplacianGP, ProductLaplacianGP)):
        model.covar_module.outputscale = torch.ones_like(model.covar_module.outputscale)
    elif isinstance(model, (AdditiveDiffusionGP, AdditiveLaplacianGP)):
        # TODO : keep previous ratio is better?
        model.covar_module.kernels[0].outputscale = torch.ones_like(model.covar_module.kernels[0].outputscale)
        model.covar_module.kernels[1].outputscale = torch.ones_like(model.covar_module.kernels[1].outputscale)
    if sample is not None:
        for k, v in sample.items():
            if 'covar_module' in k and 'outputscale' not in k:
                obj = model
                for attr in k.split('.')[:-1]:
                    obj = getattr(obj, attr)
                p_name = k.split('.')[-1].replace('_prior', '')
                setattr(obj, p_name, v.view(getattr(obj, p_name).size()))
    n_train = train_x.size(0)
    gram_matrix = model.covar_module(train_x, train_x).evaluate().detach()
    gram_matrix += torch.eye(n_train) * model.likelihood.noise.detach()
    quadratic_term = torch.matmul(train_y.t(),
                                  torch.lstsq(train_y, gram_matrix)[0]).clamp(min=torch.sum(train_y ** 2).item() * 1e-8)
    most_likely_outputscale = 2 / n_train * quadratic_term
    log_most_likely_outputscale = torch.log(most_likely_outputscale).item()
    return LogNormalPrior(loc=log_most_likely_outputscale, scale=3)


# ====================================================================================================================
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# MOST OF PARAMETERS RELATED TO BAYESIAN OPTIMIZATION ARE COLLECTED ABOVE, so only above can be modified for EXP
# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================

class CombinatorialGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor):
        noisevar_prior = kernel_parameter_prior_noisevar(train_x, train_y)
        constmean_prior = kernel_parameter_prior_constmean(train_x, train_y)
        likelihood = GaussianLikelihood(noise_prior=noisevar_prior, noise_constraint=Positive())
        self._set_dimensions(train_X=train_x, train_Y=train_y)
        train_x, train_y, _ = self._transform_tensor_args(X=train_x, Y=train_y)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=constmean_prior)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class LaplacianGP(CombinatorialGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        super().__init__(train_x=train_x, train_y=train_y)
        assert n_continuous == 0
        n_variables = len(fourier_freq)
        self.covar_module = ScaleKernel(
            ARDRegularizedLaplacianKernel(
                fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                beta_prior=kernel_parameter_prior_beta_laplacian(n_variables), beta_constraint=Positive(),
                active_dims=torch.tensor(range(n_variables))),
            outputscale_prior=outputscale_prior, outputscale_constraint=Positive())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class DiffusionGP(CombinatorialGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        super().__init__(train_x=train_x, train_y=train_y)
        assert n_continuous == 0
        n_variables = len(fourier_freq)
        self.covar_module = ScaleKernel(
            ARDDiffusionKernel(
                fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                beta_prior=kernel_parameter_prior_beta_diffusion(n_variables), beta_constraint=Positive(),
                active_dims=torch.tensor(range(n_variables))),
            outputscale_prior=outputscale_prior, outputscale_constraint=Positive())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class MixedVariableGP(ExactGP):
    def __init__(self, n_continuous: int, n_discrete: int, train_x: Optional[Tensor], train_y: Optional[Tensor]):
        self.n_continuous = n_continuous
        self.n_discrete = n_discrete
        assert (train_x is None) == (train_y is None)
        noisevar_prior = kernel_parameter_prior_noisevar(train_x, train_y)
        constmean_prior = kernel_parameter_prior_constmean(train_x, train_y)
        likelihood = GaussianLikelihood(noise_prior=noisevar_prior, noise_constraint=Positive())
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=constmean_prior)
        self.covar_module = None

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    def init_params(self):
        y_mean = 0 if self.train_targets is None else torch.mean(self.train_targets).item()
        y_var = 1 if self.train_targets is None else torch.var(self.train_targets).item()
        self.mean_module.constant.data.normal_(mean=y_mean, std=y_var ** 0.5 * 0.1)
        self.likelihood.noise = torch.exp(torch.empty_like(self.likelihood.noise_covar.noise).uniform_(-4, -2)) * y_var
        lognormal_mean = math.log(y_var * 0.9999)
        lognormal_std = (2 * (math.log(y_var) - lognormal_mean)) ** 0.5
        return lognormal_mean, lognormal_std

    def init_covar_params(self):
        tensor_like = self.likelihood.noise_covar.noise.data
        ls = tensor_like.new_zeros(self.n_continuous).uniform_(0.25, 2.0)
        beta = tensor_like.new_zeros(self.n_discrete).uniform_(0.0, 2.0)
        alpha = tensor_like.new_zeros(self.n_discrete).uniform_(0.0, 2.0)
        return ls, beta, alpha


class ModulatedLaplacianGP(MixedVariableGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        n_discrete = len(fourier_freq)
        super().__init__(n_continuous, n_discrete, train_x, train_y)
        alpha_prior = kernel_parameter_prior_alpha_laplacian(n_discrete)
        beta_prior = kernel_parameter_prior_beta_laplacian(n_discrete)
        modulatorls_prior = kernel_parameter_prior_lengthscale(n_continuous)

        covar = ModulatedRegularizedLaplacianKernel(
            n_modulators=n_continuous, fourier_freq=fourier_freq, fourier_basis=fourier_basis,
            alpha_prior=alpha_prior, alpha_constraint=Positive(),
            beta_prior=beta_prior, beta_constraint=Positive(),
            modulatorls_prior=modulatorls_prior, modulatorls_constraint=Positive())
        self.covar_module = ScaleKernel(covar, outputscale_prior=outputscale_prior, outputscale_constraint=Positive())

    def init_params(self):
        lognormal_mean, lognormal_std = super(ModulatedLaplacianGP, self).init_params()
        self.covar_module.outputscale = torch.empty_like(self.covar_module.outputscale).log_normal_(
            lognormal_mean, lognormal_std)
        ls, beta, alpha = self.init_covar_params()
        self.covar_module.base_kernel.modulatorls = ls
        self.covar_module.base_kernel.beta = beta
        self.covar_module.base_kernel.alpha = alpha


class ProductLaplacianGP(MixedVariableGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        n_discrete = len(fourier_freq)
        super().__init__(n_continuous, n_discrete, train_x, train_y)
        beta_prior = kernel_parameter_prior_beta_laplacian(n_discrete)
        lengthscale_prior = kernel_parameter_prior_lengthscale(n_continuous)

        covar_continuous = RBFKernel(
            ard_num_dims=n_continuous,
            lengthscale_prior=lengthscale_prior, lengthscale_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous)))
        covar_variable = ARDRegularizedLaplacianKernel(
            fourier_freq=fourier_freq, fourier_basis=fourier_basis,
            beta_prior=beta_prior, beta_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous, n_continuous + len(fourier_freq))))
        self.covar_module = ScaleKernel(
            ProductKernel(covar_continuous, covar_variable),
            outputscale_prior=outputscale_prior, outputscale_constraint=Positive())
        self.covar_module.n_continuous = n_continuous

    def init_params(self):
        lognormal_mean, lognormal_std = super().init_params()
        self.covar_module.outputscale = torch.empty_like(self.covar_module.outputscale).log_normal_(
            lognormal_mean, lognormal_std)
        ls, beta, _ = self.init_covar_params()
        self.covar_module.base_kernel.kernels[0].lengthscale = ls
        self.covar_module.base_kernel.kernels[1].beta = beta


class AdditiveLaplacianGP(MixedVariableGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        n_discrete = len(fourier_freq)
        super().__init__(n_continuous, n_discrete, train_x, train_y)
        beta_prior = kernel_parameter_prior_beta_laplacian(n_discrete)
        lengthscale_prior = kernel_parameter_prior_lengthscale(n_continuous)

        covar_continuous = ScaleKernel(
            RBFKernel(
                ard_num_dims=n_continuous,
                lengthscale_prior=lengthscale_prior, lengthscale_constraint=Positive()),
            outputscale_prior=outputscale_prior, outputscale_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous)))
        covar_variable = ScaleKernel(
            ARDRegularizedLaplacianKernel(
                fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                beta_prior=beta_prior, beta_constraint=Positive()),
            outputscale_prior=deepcopy(outputscale_prior), outputscale_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous, n_continuous + len(fourier_freq))))
        self.covar_module = AdditiveKernel(covar_continuous, covar_variable)
        self.covar_module.n_continuous = n_continuous

    def init_params(self):
        lognormal_mean, lognormal_std = super().init_params()
        outputscale = torch.empty_like(self.covar_module.kernels[0].outputscale).\
            log_normal_(lognormal_mean, lognormal_std)
        self.covar_module.kernels[0].outputscale = outputscale / 2.0
        self.covar_module.kernels[1].outputscale = outputscale / 2.0
        ls, beta, _ = self.init_covar_params()
        self.covar_module.kernels[0].base_kernel.lengthscale = ls
        self.covar_module.kernels[1].base_kernel.beta = beta


class ModulatedDiffusionGP(MixedVariableGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        n_discrete = len(fourier_freq)
        super().__init__(n_continuous, n_discrete, train_x, train_y)
        alpha_prior = kernel_parameter_prior_alpha_diffusion(n_discrete)
        beta_prior = kernel_parameter_prior_beta_diffusion(n_discrete)
        modulatorls_prior = kernel_parameter_prior_lengthscale(n_continuous)

        covar = ModulatedDiffusionKernel(
            n_modulators=n_continuous, fourier_freq=fourier_freq, fourier_basis=fourier_basis,
            alpha_prior=alpha_prior, alpha_constraint=Positive(),
            beta_prior=beta_prior, beta_constraint=Positive(),
            modulatorls_prior=modulatorls_prior, modulatorls_constraint=Positive())
        self.covar_module = ScaleKernel(
            covar, outputscale_prior=outputscale_prior, outputscale_constraint=Positive())

    def init_params(self):
        lognormal_mean, lognormal_std = super().init_params()
        self.covar_module.outputscale = torch.empty_like(self.covar_module.outputscale).log_normal_(
            lognormal_mean, lognormal_std)
        ls, beta, alpha = self.init_covar_params()
        self.covar_module.base_kernel.modulatorls = ls
        self.covar_module.base_kernel.beta = beta
        self.covar_module.base_kernel.alpha = alpha


class ProductDiffusionGP(MixedVariableGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        n_discrete = len(fourier_freq)
        super().__init__(n_continuous, n_discrete, train_x, train_y)
        beta_prior = kernel_parameter_prior_beta_diffusion(n_discrete)
        lengthscale_prior = kernel_parameter_prior_lengthscale(n_continuous)

        covar_continuous = RBFKernel(
            ard_num_dims=n_continuous,
            lengthscale_prior=lengthscale_prior, lengthscale_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous)))
        covar_variable = ARDDiffusionKernel(
            fourier_freq=fourier_freq, fourier_basis=fourier_basis,
            beta_prior=beta_prior, beta_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous, n_continuous + len(fourier_freq))))
        self.covar_module = ScaleKernel(
            ProductKernel(covar_continuous, covar_variable),
            outputscale_prior=outputscale_prior, outputscale_constraint=Positive())
        self.covar_module.n_continuous = n_continuous

    def init_params(self):
        lognormal_mean, lognormal_std = super().init_params()
        self.covar_module.outputscale = torch.empty_like(self.covar_module.outputscale).log_normal_(
                lognormal_mean, lognormal_std)
        ls, beta, _ = self.init_covar_params()
        self.covar_module.base_kernel.kernels[0].lengthscale = ls
        self.covar_module.base_kernel.kernels[1].beta = beta


class AdditiveDiffusionGP(MixedVariableGP):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 train_x: Tensor, train_y: Tensor, outputscale_prior: Optional[Prior] = None):
        n_discrete = len(fourier_freq)
        super().__init__(n_continuous, n_discrete, train_x, train_y)
        beta_prior = kernel_parameter_prior_beta_diffusion(n_discrete)
        lengthscale_prior = kernel_parameter_prior_lengthscale(n_continuous)

        covar_continuous = ScaleKernel(
            RBFKernel(
                ard_num_dims=n_continuous,
                lengthscale_prior=lengthscale_prior, lengthscale_constraint=Positive()),
            outputscale_prior=outputscale_prior, outputscale_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous)))
        covar_variable = ScaleKernel(
            ARDDiffusionKernel(
                fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                beta_prior=beta_prior, beta_constraint=Positive()),
            outputscale_prior=deepcopy(outputscale_prior), outputscale_constraint=Positive(),
            active_dims=torch.tensor(range(n_continuous, n_continuous + len(fourier_freq))))
        self.covar_module = AdditiveKernel(covar_continuous, covar_variable)
        self.covar_module.n_continuous = n_continuous

    def init_params(self):
        lognormal_mean, lognormal_std = super().init_params()
        outputscale = torch.empty_like(self.covar_module.kernels[0].outputscale).\
            log_normal_(lognormal_mean, lognormal_std)
        self.covar_module.kernels[0].outputscale = outputscale / 2.0
        self.covar_module.kernels[1].outputscale = outputscale / 2.0
        ls, beta, _ = self.init_covar_params()
        self.covar_module.kernels[0].base_kernel.lengthscale = ls
        self.covar_module.kernels[1].base_kernel.beta = beta


class MixedVariableGPBase(object):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor], kernel_type: str):
        super().__init__()
        assert kernel_type in KERNEL_TYPE_LIST

        self.gp_constructor = globals()[kernel_type + 'GP']
        self._kernel_type = kernel_type

        self._n_continuous = n_continuous
        self._fourier_freq = fourier_freq
        self._fourier_basis = fourier_basis

    def state_dict(self):
        return {'n_continuous': self._n_continuous,
                'fourier_freq': self._fourier_freq, 'fourier_basis': self._fourier_basis,
                'kernel_type': self._kernel_type}


class MixedVariableGPOptimize(MixedVariableGPBase):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor], kernel_type: str):
        super().__init__(n_continuous=n_continuous, fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                         kernel_type=kernel_type)
        self.info_str = 'MV-K%s-Optimize' % kernel_type
        self.gp_model = self.gp_constructor(
            n_continuous=self._n_continuous, fourier_freq=self._fourier_freq, fourier_basis=self._fourier_basis,
            train_x=None, train_y=None, outputscale_prior=kernel_parameter_prior_outputscale(None, None, None, None))
        self.gp_model.init_params()

        self.n_runs: int = OPTIMIZE_N_RUNS

    def state_dict(self):
        states = deepcopy(super().state_dict())
        states.update({'n_runs': self.n_runs})
        return states

    def optimize(self, train_x: Tensor, train_y: Tensor):
        train_y = train_y.view(-1)
        self.gp_model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        n_processes = max(multiprocessing.cpu_count() // int(OPTIMIZE_N_CORES_USED), 1)
        pool = multiprocessing.Pool(n_processes)
        args_list = []
        for n in range(self.n_runs):
            model = deepcopy(self.gp_model)
            if n > 0:
                model.init_params()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            args_list.append((train_x, train_y, model, mll, optimizer, n + 1))

        start_time = time.time()
        print('Negative Marginal Likelihood Minimization : '
              'Multiprocessing with %d processes for %d Random initialization' % (n_processes, self.n_runs))
        instance_id, models, nmlls = list(zip(*pool.starmap_async(optimize_mll, args_list).get()))
        model_dict = dict(zip(instance_id, models))
        negative_mll = dict(zip(instance_id, nmlls))
        print('%12d seconds to run %d negative marginal lilkelihood minimization' %
              (time.time() - start_time, self.n_runs))

        best_model_name = pick_best_model(negative_mll, model_dict)
        print('ID:%2d with %+.6f has been chosen' % (best_model_name, negative_mll[best_model_name]))
        self.gp_model = model_dict[best_model_name]


def optimize_mll(train_x, train_y, model, mll, optimizer, instance_id: int):
    model_name = '%02d' % instance_id
    model.train()

    start_time = time.time()
    prev_loss = float('inf')
    prev_model_state_dict = deepcopy(model.state_dict())
    i = 0
    while True:
        optimizer.zero_grad()
        output = model(train_x)
        try:
            loss = -mll(output, train_y)
        except gpytorch.utils.errors.NotPSDError:
            model.load_state_dict(prev_model_state_dict)
            loss = prev_loss * torch.ones(1)
            print('        %s gpytorch.utils.errors.NotPSDError occurred' % model_name)
            sys.stdout.flush()
            break
        prev_model_state_dict = deepcopy(model.state_dict())
        loss.backward()
        optimizer.step()
        # ftol of scipy.optimize.minimize(method='L-BFGS-B')
        if abs((prev_loss - loss.item()) / max(abs(prev_loss), 1)) < OPTIMIZE_REL_TOL:
            break
        prev_loss = loss.item()
        i += 1
    print('    %s : %6d updates - %5d seconds, loss : %+.8f, noise : %.6f' %
          (model_name, i, int(time.time() - start_time), loss.item(), model.likelihood.noise.item()))
    sys.stdout.flush()
    return instance_id, model, loss.item()


def pick_best_model(negative_mll: Dict, model_dict: Dict) -> str:
    best_model_name = min(negative_mll, key=negative_mll.get)

    info_str_list = []
    for instance_id, model in model_dict.items():
        if isinstance(model.covar_module, ScaleKernel):
            outputscale = model.covar_module.outputscale.data.item()
        elif isinstance(model.covar_module, AdditiveKernel):
            outputscale = sum([elm.outputscale.data.item() for elm in model.covar_module.kernels])
        else:
            raise NotImplementedError
        info_str_list.append('%2d neg.mll : %12.8f / outputscale : %12.8f / noise : %12.8f / ratio : %12.2f %s'
                             % (instance_id, negative_mll[instance_id], outputscale, model.likelihood.noise.data.item(),
                                outputscale / model.likelihood.noise.data.item(), '++'
                                if best_model_name == instance_id else ''))
    print('\n'.join(info_str_list))

    return best_model_name


class MixedVariableGPMCMC(MixedVariableGPBase):
    def __init__(self, n_continuous: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 kernel_type: str, sampler_type: str, num_samples: int = 10, warmup_steps: int = 0,
                 mcmc_step_size: Optional[float] = None):
        super().__init__(n_continuous=n_continuous, fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                         kernel_type=kernel_type)
        self.info_str = 'MV-K%s-S%s' % (kernel_type, sampler_type)
        assert sampler_type in SAMPLER_TYPE_LIST

        self._model_samples: Optional[Dict[str, Tensor]] = None
        self._last_sample_unconstrained: Optional[Dict[str, Tensor]] = None
        self._last_sample_original: Optional[Dict[str, Tensor]] = None
        self._num_samples = num_samples
        self._warmup_steps = warmup_steps
        self._mcmc_step_size = mcmc_step_size
        self._sampler_type = sampler_type

    def state_dict(self):
        states = deepcopy(super().state_dict())
        states.update(
            {'sampler_type': self._sampler_type, 'num_samples': self._num_samples, 'warmup_steps': self._warmup_steps,
             'last_sample_original': self._last_sample_original,
             'last_sample_unconstrained': self._last_sample_unconstrained, 'mcmc_step_size': self._mcmc_step_size})
        return states

    @property
    def model_samples(self):
        return self._model_samples

    @property
    def last_sample_unconstrained(self):
        return self._last_sample_unconstrained

    @last_sample_unconstrained.setter
    def last_sample_unconstrained(self, sample: Dict[str, Tensor]):
        self._last_sample_unconstrained = sample

    @property
    def last_sample_original(self):
        return self._last_sample_original

    @last_sample_original.setter
    def last_sample_original(self, sample: Dict[str, Tensor]):
        self._last_sample_original = sample

    def _adaptive_outputscale_prior(self, train_x: Tensor, train_y: Tensor) -> Prior:
        model = self.gp_constructor(n_continuous=self._n_continuous,
                                    fourier_freq=self._fourier_freq, fourier_basis=self._fourier_basis,
                                    train_x=train_x, train_y=train_y, outputscale_prior=Prior())  # dummy Prior

        return kernel_parameter_prior_outputscale(model=model, sample=self._last_sample_original,
                                                  train_x=train_x, train_y=train_y)

    @staticmethod
    def copy_from_pyro_sample(model, sample):
        for elm in model.named_priors():
            name, prior, closure, setting_closure = elm
            param_size = closure().size()
            setting_closure(sample[name].view(param_size))

    def sample(self, train_x: Tensor, train_y: Tensor):
        outputscale_prior = self._adaptive_outputscale_prior(train_x, train_y)
        model = self.gp_constructor(n_continuous=self._n_continuous,
                                    fourier_freq=self._fourier_freq, fourier_basis=self._fourier_basis,
                                    train_x=train_x, train_y=train_y, outputscale_prior=outputscale_prior)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # TODO : Should below be Multivariate Gaussian likelihood using training data?
        def pyro_model(x, y):
            model.pyro_sample_from_prior()
            output = model(x)
            loss = mll.pyro_factor(output, y)
            return y

        if self._sampler_type == 'NUTS':
            mcmc_kernel = NUTS(pyro_model, step_size=self._mcmc_step_size, adapt_step_size=True,
                               adapt_mass_matrix=False)
        elif self._sampler_type == 'Slice_groupwise':
            mcmc_kernel = Slice(pyro_model, grouping="groupwise")
        elif self._sampler_type == 'Slice_elementwise':
            mcmc_kernel = Slice(pyro_model, grouping="elementwise", shuffle_type='elementwise')
        else:
            raise NotImplementedError
        if self._last_sample_unconstrained is None:
            mcmc = MCMC(mcmc_kernel,
                        num_samples=1, warmup_steps=self._warmup_steps,
                        disable_progbar=False)
            mcmc.run(train_x, train_y)
            samples = mcmc.get_samples()
        else:
            mcmc = MCMC(mcmc_kernel, initial_params=self._last_sample_unconstrained,
                        num_samples=self._num_samples, warmup_steps=self._warmup_steps,
                        disable_progbar=False)
            mcmc.run(train_x, train_y)
            samples = mcmc.get_samples()
            self._model_samples = []
            for s in range(self._num_samples):
                # deepcopy is not working well, so simply instantiating the same models resolve the issue
                model = self.gp_constructor(n_continuous=self._n_continuous,
                                            fourier_freq=self._fourier_freq, fourier_basis=self._fourier_basis,
                                            train_x=train_x, train_y=train_y, outputscale_prior=outputscale_prior)
                self.copy_from_pyro_sample(model, {key: value[s] for key, value in samples.items()})
                self._model_samples.append(model)

        if isinstance(mcmc_kernel, HMC):
            self._mcmc_step_size = mcmc_kernel.step_size

        self._last_sample_original = {key: value[-1] for key, value in samples.items()}
        self._last_sample_unconstrained = {key: mcmc.transforms[key](value[-1]) for key, value in samples.items()}
