from typing import Callable

import torch
from torch import Tensor
from torch.distributions import Normal

from gpytorch.utils.cholesky import psd_safe_cholesky

from FrequencyModulatedKernelBO.experiments.gp_models import MixedVariableGPMCMC, MixedVariableGPOptimize


class AcquisitionFunction(object):
    """
    In the original BoTorch implementation, acquisition function performs unnecessary repetition of gram matrix
    computation, in this class, such unnecessary repeated computation is replaced with cached values,
    e.g. cholesky decomposition of gram matrix
    This speeds up acquisition function optimization significantly
    """
    def __init__(self, surrogate: MixedVariableGPOptimize, acq_function: Callable,
                 data_x: Tensor, data_y: Tensor, minimize: bool):
        self.gp_model = surrogate.gp_model
        self.data_x = data_x
        self.data_y = data_y
        self.minimize = minimize
        # In acq. func. optimization, optimization is performed w.r.t input not w.r.t parameter. detach() is OK
        gram_matrix = self.gp_model.covar_module(data_x).evaluate().detach() \
                      + torch.eye(data_x.size(0), device=data_x.device) * self.gp_model.likelihood.noise.detach()
        self.cholesky_factor_lower = psd_safe_cholesky(gram_matrix, upper=False).detach()
        self.mean_train = self.gp_model.mean_module(self.data_x).detach()
        self.acq_function = acq_function
        self.acq_func_kwargs = {}

    def __call__(self, x):
        acq_func_value_list = []
        k_train_test = self.gp_model.covar_module(self.data_x, x).evaluate()
        k_test_test_diag = self.gp_model.covar_module(x, diag=True)
        mean_test = self.gp_model.mean_module(x)
        chol_solve = torch.triangular_solve(
            torch.cat([k_train_test, self.data_y.view(-1, 1) - self.mean_train.view(-1, 1)], dim=1),
            self.cholesky_factor_lower, upper=False)[0]
        pred_mean = torch.matmul(chol_solve[:, -1:].t(), chol_solve[:, :-1]).view(-1) + mean_test.view(-1)
        pred_std = (k_test_test_diag.view(-1) - torch.sum(chol_solve[:, :-1] ** 2, dim=0)).clamp(min=1e-9) ** 0.5
        acq_func_value_list.append(self.acq_function(mean=pred_mean, sigma=pred_std, minimize=self.minimize,
                                                     **self.acq_func_kwargs))
        return torch.stack(acq_func_value_list, dim=0).mean(dim=0)


class AverageAcquisitionFunction(object):
    """
    In the original BoTorch implementation, acquisition function performs unnecessary repetition of gram matrix
    computation, in this class, such unnecessary repeated computation is replaced with cached values,
    e.g. cholesky decomposition of gram matrix
    This speeds up acquisition function optimization significantly
    """
    def __init__(self, surrogate: MixedVariableGPMCMC, acq_function: Callable,
                 data_x: Tensor, data_y: Tensor, minimize: bool):
        self.gp_models = {i: gp_model for i, gp_model in enumerate(surrogate.model_samples)}
        self.data_x = data_x
        self.data_y = data_y
        self.minimize = minimize
        # In acq. func. optimization, optimization is performed w.r.t input not w.r.t parameter. detach() is OK
        self.cholesky_factor_lower = dict()
        self.mean_train = dict()
        for i, gp_model in self.gp_models.items():
            gram_matrix = gp_model.covar_module(data_x).evaluate().detach() \
                          + torch.eye(data_x.size(0), device=data_x.device) * gp_model.likelihood.noise.detach()
            self.cholesky_factor_lower[i] = psd_safe_cholesky(gram_matrix, upper=False).detach()
            self.mean_train[i] = gp_model.mean_module(self.data_x).detach()
        self.acq_function = acq_function
        self.acq_func_kwargs = {}

    def __call__(self, x):
        acq_func_value_list = []
        for i, gp_model in self.gp_models.items():
            k_train_test = gp_model.covar_module(self.data_x, x).evaluate()
            k_test_test_diag = gp_model.covar_module(x, diag=True)
            mean_test = gp_model.mean_module(x)
            chol_solve = torch.triangular_solve(
                torch.cat([k_train_test, self.data_y.view(-1, 1) - self.mean_train[i].view(-1, 1)], dim=1),
                self.cholesky_factor_lower[i], upper=False)[0]
            pred_mean = torch.matmul(chol_solve[:, -1:].t(), chol_solve[:, :-1]).view(-1) + mean_test.view(-1)
            pred_std = (k_test_test_diag.view(-1) - torch.sum(chol_solve[:, :-1] ** 2, dim=0)).clamp(min=1e-9) ** 0.5
            acq_func_value_list.append(self.acq_function(mean=pred_mean, sigma=pred_std, minimize=self.minimize,
                                                         **self.acq_func_kwargs))
        return torch.stack(acq_func_value_list, dim=0).mean(dim=0)


def expected_improvement_mean_sigma(mean, sigma, incumbent: float, minimize: bool = True) -> Tensor:
    u = (mean - incumbent) / sigma
    if minimize:
        u = -u
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei


def optimization_as_estimation(mean, sigma, incumbent: float, minimize: bool = True) -> Tensor:
    if not minimize:
        return (mean - incumbent) / sigma
    else:
        return (incumbent - mean) / sigma


def numerical_integration(mean: Tensor, sigma: Tensor, m0: float, dw: float = 0.005):
    # Assuming estimating maximum
    normal = Normal(torch.zeros_like(mean), torch.ones_like(mean))
    w = m0
    m_hat = m0
    prev_logprodphi = -1  # any negative value is OK to begin with
    while prev_logprodphi < 0:  # logprodphi becomes NUMERICALLY zero due to numerical capability, this stops.
        logprodphi = torch.sum(torch.log(normal.cdf((w - mean) / sigma)))
        m_hat = m_hat + (1 - torch.exp(logprodphi)).item() * dw
        w += dw
        prev_logprodphi = logprodphi
    maximum_est = m_hat
    return maximum_est


def m_hat_estimate(gp_model, data_x, data_y, normalized_input_info,
                   mean_train, cholesky_lower_inv, maximize: bool = False) -> float:
    """

    :param gp_model:
    :param data_x:
    :param data_y:
    :param mean_train:
    :param cholesky_lower_inv:
    :param maximize:
    :return:
    """
    n_samples = 100000
    n_data, n_dim = data_x.size()
    x = data_x.new_empty((n_samples, n_dim))
    x[:n_data] = data_x
    for d in range(n_dim):
        if isinstance(normalized_input_info[d], tuple):
            lower, upper = normalized_input_info[d]
            x[n_data:n_samples, d] = torch.rand_like(x[n_data:n_samples, d]) * (upper - lower) + lower
        elif isinstance(normalized_input_info[d], Tensor):
            x[n_data:n_samples, d] = torch.randint_like(x[n_data:n_samples, d], high=normalized_input_info[d].size()[0])
        else:
            raise ValueError
    mean, sigma = pred_mean_std(x=x, gp_model=gp_model, data_x=data_x, data_y=data_y,
                                mean_train=mean_train, cholesky_lower_inv=cholesky_lower_inv)

    mean = mean * (1 if maximize else -1)
    best_f = torch.max(data_y * (1 if maximize else -1)).item()

    maximum_est = numerical_integration(mean=mean, sigma=sigma, m0=best_f, dw=0.005)

    return maximum_est * (1 if maximize else -1)


def pred_mean_std(x: Tensor, gp_model, data_x: Tensor, data_y: Tensor,
                  mean_train: Tensor, cholesky_lower_inv: Tensor):
    """

    :param x: points where prediction is made
    :param gp_model:
    :param data_x:
    :param data_y:
    :param mean_train: Cached for faster computation
    :param cholesky_lower_inv: Cached for faster computation
    :return:
    """
    # backprogpagation is NOT called via this class, so using detach() is fine !!
    k_train_test = gp_model.covar_module(data_x, x).evaluate()
    k_test_test_diag = gp_model.covar_module(x, diag=True).detach()
    mean_test = gp_model.mean_module(x).detach()
    chol_solve = torch.mm(cholesky_lower_inv,
                          torch.cat([k_train_test, data_y.view(-1, 1) - mean_train.view(-1, 1)], dim=1))
    pred_mean = torch.mm(chol_solve[:, -1:].t(), chol_solve[:, :-1]).view(-1) + mean_test.view(-1)
    pred_std = (k_test_test_diag.view(-1) - torch.sum(chol_solve[:, :-1] ** 2, dim=0)).clamp(min=1e-9) ** 0.5
    return pred_mean, pred_std


def gram_cholesky_lower_inv(gp_model, data_x: Tensor) -> Tensor:
    gram_matrix = gp_model.covar_module(data_x).evaluate().detach() \
                  + torch.eye(data_x.size()[0], device=data_x.device) * gp_model.likelihood.noise.detach()
    cholesky_lower = psd_safe_cholesky(gram_matrix, upper=False)
    cholesky_lower_inv = torch.triangular_solve(
        input=torch.eye(gram_matrix.size(0), device=data_x.device), A=cholesky_lower, upper=False)[0]
    return cholesky_lower_inv