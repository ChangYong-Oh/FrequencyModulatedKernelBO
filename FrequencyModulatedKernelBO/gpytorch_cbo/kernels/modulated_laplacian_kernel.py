from typing import Optional, List

import torch
from torch import Tensor
from gpytorch.priors import Prior
from gpytorch.constraints import Positive, Interval

from FrequencyModulatedKernelBO.gpytorch_cbo.kernels.ard_laplacian_kernel import ARDRegularizedLaplacianKernel


class ModulatedRegularizedLaplacianKernel(ARDRegularizedLaplacianKernel):
    r"""
    Computes a covariance matrix based the ARD diffusion kernel
    (Combinatorial Bayesian Optimization using the Graph Cartesian Product)
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    "Eigenvalue bounds for symmetric matrices with entries in one interval" Huinan Leng, Zhiqing He
    To find a lower bound of the smallest eigenvalue (negative) of a real symmetric matrix

    Perron–Frobenius theorem to find upper and lower bound of the largest eigenvalue of positive matrix

    "ON THE OPTIMALITY AND SHARPNESS OF LAGUERRE’S  LOWER BOUND ON THE SMALLEST EIGENVALUE OF A SYMMETRIC POSITIVE
    DEFINITE MATRIX" Yusaku Yamamoto
    To find a lower bound of the smallest eigenvalue (positive) of a positive definite matrix

    if self.batch_size.numel() > 1
    All batches use the same fourier_frequency and fourier_basis

    .. math::

    where

    .. note::

    """
    def __init__(self, n_modulators: int, fourier_freq: List[Tensor], fourier_basis: List[Tensor],
                 alpha_prior: Optional[Prior] = None, alpha_constraint: Optional[Positive] = Positive(),
                 beta_prior: Optional[Prior] = None, beta_constraint: Optional[Positive] = Positive(),
                 modulatorls_prior: Optional[Prior] = None, modulatorls_constraint: Optional[Positive] = Positive(),
                 **kwargs):
        super().__init__(fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                         beta_prior=beta_prior, beta_constraint=beta_constraint,
                         **kwargs)
        self.n_modulators = n_modulators

        self.register_parameter(name="raw_alpha",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.n_variables)))
        self.register_parameter(name="raw_modulatorls",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.n_modulators)))
        # In raw_modulatorls, the inserted 1 is to accomondate broadcasting when batch_shape.numel() > 1

        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, lambda: self.alpha, lambda v: self._set_alpha(v))
        if modulatorls_prior is not None:
            self.register_prior("modulatorls_prior", modulatorls_prior,
                                lambda: self.modulatorls, lambda v: self._set_modulatorls(v))

        self.register_constraint("raw_alpha", alpha_constraint)
        self.register_constraint("raw_modulatorls", modulatorls_constraint)

        self.register_buffer('modulator_dist_sq', None)

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value: torch.Tensor) -> None:
        self._set_alpha(value)

    def _set_alpha(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def modulatorls(self) -> torch.Tensor:
        return self.raw_modulatorls_constraint.transform(self.raw_modulatorls)

    @modulatorls.setter
    def modulatorls(self, value: torch.Tensor) -> None:
        self._set_modulatorls(value)

    def _set_modulatorls(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_modulatorls)
        self.initialize(raw_modulatorls=self.raw_modulatorls_constraint.inverse_transform(value))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                diag: Optional[bool] = False, ** params) -> torch.Tensor:
        assert x1.size(-1) == x2.size(-1) == self.n_modulators + self.n_variables

        x1_c = torch.narrow(x1, -1, 0, self.n_modulators)
        x1_v = torch.narrow(x1, -1, self.n_modulators, self.n_variables).long()
        x2_c = torch.narrow(x2, -1, 0, self.n_modulators)
        x2_v = torch.narrow(x2, -1, self.n_modulators, self.n_variables).long()

        # context_dist_sq : b X n1 X n2 if diag = False else b X (n1 = n2)
        self.modulator_dist_sq = self.covar_dist(x1_c.div(self.modulatorls), x2_c.div(self.modulatorls),
                                                 diag=diag, square_dist=True)
        modulator_dist_sq = self.modulator_dist_sq / self.n_modulators
        modulatorls_inv_sq_mean = torch.mean((1.0 / torch.narrow(self.modulatorls, -1, 0, 1) ** 2), dim=-1)

        gram_full = 1
        for n in range(self.n_variables):
            # alpha_n : b X 1
            alpha_n = torch.narrow(self.alpha, -1, n, 1)
            # beta_n : b X 1
            beta_n = torch.narrow(self.beta, -1, n, 1)
            # freq_regularizer : b X |V|
            freq_regularizer = 1.0 / (1 + beta_n * self.fourier_freq[n]
                                      + alpha_n * modulatorls_inv_sq_mean / self.n_variables)
            # basis_x1_v : b X n1 X |V| / basis_x2_v : b X n2 X |V|
            basis_x1_v = self.fourier_basis[n][torch.narrow(x1_v, -1, n, 1).squeeze(-1).long()]
            basis_x2_v = self.fourier_basis[n][torch.narrow(x2_v, -1, n, 1).squeeze(-1).long()]
            if diag:
                # alpha_n * normalized_dist_sq : b X (n1 = n2) / self.fourier_freq[n] : |V|
                # modulated_freq_regularizer : tensor of size : b X (n1 = n2) X |V|
                modulated_freq = 1.0 / (1 + (beta_n * self.fourier_freq[n]).unsqueeze(-2)
                                        + (alpha_n * modulator_dist_sq / self.n_variables).unsqueeze(-1))
                gram_factor = (basis_x1_v * modulated_freq * basis_x2_v).sum(dim=-1)
            else:
                # alpha_n * normalized_dist_sq : b X n1 X n2 / self.fourier_freq[n] : |V|
                # modulated_freq_regularizer : tensor of size : b X n1 X n2 X |V|
                modulated_freq = 1.0 / (1 + (beta_n * self.fourier_freq[n]).unsqueeze(-2).unsqueeze(-2)
                                        + (alpha_n * modulator_dist_sq / self.n_variables).unsqueeze(-1))
                gram_factor = (basis_x1_v.unsqueeze(-2) * modulated_freq * basis_x2_v.unsqueeze(-3)).sum(-1)
            # division by torch.mean(freq_regularizer) is for numerical stability,
            # this is compensate by outputscale parameter using gpytorch.kernels.ScaleKernel
            gram_full *= gram_factor / torch.mean(freq_regularizer).view(1, 1)
        return gram_full.squeeze(0) if diag else gram_full


if __name__ == '__main__':
    import time
    import numpy as np
    from FrequencyModulatedKernelBO.utils.random_data import random_laplacian, random_data

    _n_modulators = 8
    _same_context = False

    _b = 0
    _n_variables = 12
    _n_data = 200

    _rnd_seed = None

    _n_runs = 500
    _cnt = 0
    _kernel_eval_time = 0
    for _r in range(_n_runs):
        _n_data = (_r % 196) + 105
        _seed_list = np.random.RandomState(_rnd_seed).randint(0, 1000000, 10)

        _batch_size = torch.Size([_b]) if _b > 0 else torch.Size([])
        _n_choices = list(np.random.RandomState(_seed_list[0]).randint(2, 10, _n_variables))
        _fourier_freq, _fourier_basis = random_laplacian(_n_choices, _seed_list[1])
        _kernel = ModulatedRegularizedLaplacianKernel(_n_modulators, _fourier_freq, _fourier_basis, batch_shape=_batch_size)
        _kernel.beta = torch.from_numpy(np.random.RandomState(_seed_list[2]).uniform(0, 4, _kernel.beta.numel())).view_as(_kernel.beta)
        _x, _ = random_data(_n_modulators, _n_choices, _n_data, _seed_list[3])
        _x[:, :_n_modulators] = (_x[:, :_n_modulators] - torch.mean(_x[:, :_n_modulators], dim=0, keepdim=True)) \
                                 / torch.std(_x[:, :_n_modulators], dim=0, keepdim=True)
        _kernel.alpha = torch.from_numpy(np.random.RandomState(_seed_list[5]).uniform(0, 0.3, _kernel.alpha.numel())).view_as(_kernel.alpha)
        _kernel.modulatorls = torch.from_numpy(np.random.RandomState(_seed_list[6]).uniform(0.5, 2, _kernel.modulatorls.numel())).view_as(_kernel.modulatorls)
        if _b > 0:
            _x = _x.unsqueeze(0).repeat(_b, 1, 1)
        else:
            _start_time = time.time()
            _kernel_eval = _kernel(_x, _x, diag=False).evaluate()
            _kernel_eval_time += time.time() - _start_time
        if _b > 0:
            for _i in range(_b):
                _eigval, _ = torch.symeig(_kernel_eval[_i])
                print(_eigval)
                print(torch.min(_eigval), torch.max(_eigval))
                print(_kernel_eval[_i].size())
        else:
            _eigval, _ = torch.symeig(_kernel_eval)
            # print(_eigval)
            _eigval_min = torch.min(_eigval).item()
            _eigval_max = torch.max(_eigval).item()
            print('KerVal : %+.4f ~ %.4f / Eigval : %+.4f ~ %.4f Ratio(%5.2f%%)'
                  % (torch.min(_kernel_eval).item(), torch.max(_kernel_eval).item(),
                     _eigval_min, _eigval_max, abs(_eigval_min / _eigval_max * 100)))
        if torch.min(_eigval) < 0:
            print('eigenspectrum' + ('=' * 50))
            _eigval_min = torch.min(_eigval).item()
            _eigval_max = torch.max(_eigval).item()
            print('BMK eigval : %+.6E ~ %.6E Ratio(%5.2f%%)' % (_eigval_min, _eigval_max, abs(_eigval_min / _eigval_max * 100)))
            # _a = torch.exp(_kernel.alpha * _kernel.beta / (1 + _kernel.alpha))
            # print(_a / 2)
            _covar_dist = _kernel.covar_dist(_x[:,:_n_modulators].div(_kernel.modulatorls),
                                             _x[:,:_n_modulators].div(_kernel.modulatorls),
                                             diag=False, square_dist=True)
            _gram_mat = (torch.exp(-_covar_dist) + (_covar_dist == 0) * 1e-3)
            _eigval_ctx = torch.symeig(_gram_mat)[0]
            print('min_eig/max_eig : %.6E' % (torch.min(_eigval_ctx)/torch.max(_eigval_ctx)).item())
            _cnt += 1
    print('%d/%d On average, %f seconds to evaluate %d X %d matrix' % (_cnt, _n_runs, _kernel_eval_time / _n_runs, _n_data, _n_data))