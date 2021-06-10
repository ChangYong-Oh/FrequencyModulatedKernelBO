
from typing import Optional, Iterable

import torch
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import Positive


class ARDDiffusionKernel(Kernel):
    r"""
    Computes a covariance matrix based the ARD diffusion kernel
    (Combinatorial Bayesian Optimization using the Graph Cartesian Product)
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    if self.batch_size.numel() > 1
    All batches use the same fourier_frequency and fourier_basis

    .. math::

    where

    .. note::



    """
    def __init__(self, fourier_freq: Iterable[Tensor], fourier_basis: Iterable[Tensor],
                 beta_prior: Optional[Prior] = None,
                 beta_constraint: Optional[Positive] = None, **kwargs):
        assert len(fourier_freq) == len(fourier_basis)
        self.fourier_freq = [elm.clamp(min=0) for elm in fourier_freq]
        self.fourier_basis = fourier_basis
        self.n_variables = len(fourier_freq)
        self.n_choices = []
        for n in range(self.n_variables):
            assert self.fourier_freq[n].numel() == self.fourier_basis[n].size(0) == self.fourier_basis[n].size(1)
            self.n_choices.append(self.fourier_freq[n].numel())

        super().__init__(**kwargs)

        if beta_constraint is None:
            beta_constraint = Positive()

        self.register_parameter(name="raw_beta",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.n_variables)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        self.register_constraint("raw_beta", beta_constraint)

    @property
    def beta(self) -> torch.Tensor:
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value: torch.Tensor) -> None:
        self._set_beta(value)

    def _set_beta(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1: torch.LongTensor, x2: torch.LongTensor,
                diag: bool = False, **params) -> torch.Tensor:
        assert x1.size(-1) == x2.size(-1) == self.n_variables

        gram_full = 1
        for n in range(self.n_variables):
            # torch.narrow(self.beta, -1, n, 1) : b X 1 / self.fourier_freq[n] : |V| => freq_regularizer : b X |V|
            freq_regularizer = torch.exp(-torch.narrow(self.beta, -1, n, 1) * self.fourier_freq[n])
            basis_x = self.fourier_basis[n][torch.narrow(x1, -1, n, 1).squeeze(-1).long()] # b X n1 X |V|
            basis_x2 = self.fourier_basis[n][torch.narrow(x2, -1, n, 1).squeeze(-1).long()] # b X n2 X |V|
            if diag:
                gram_factor = (basis_x * basis_x2 * freq_regularizer.unsqueeze(-2)).sum(dim=-1)
            else:
                gram_factor = (basis_x.unsqueeze(-2) * freq_regularizer.unsqueeze(-2).unsqueeze(-2) * basis_x2.unsqueeze(-3)).sum(dim=-1)
            # division by torch.mean(freq_regularizer) is for numerical stability,
            # this is compensate by outputscale parameter using gpytorch.kernels.ScaleKernel
            gram_full *= gram_factor / torch.mean(freq_regularizer)
        return gram_full


if __name__ == '__main__':
    import numpy as np
    from FrequencyModulatedKernelBO.utils.random_data import random_laplacian, random_data

    _n_variables = 12

    _rnd_seed = 10
    _n_runs = 500

    for _r in range(_n_runs):
        _n_data = (_r % 196) + 105
        _seed_list = np.random.RandomState(None).randint(0, 1000000, 5)
        _n_choices = list(np.random.RandomState(_seed_list[0]).randint(2, 10, _n_variables))
        _fourier_freq, _fourier_basis = random_laplacian(_n_choices, _seed_list[1])
        _kernel = ARDDiffusionKernel(_fourier_freq, _fourier_basis)
        _kernel.beta = torch.from_numpy(np.random.RandomState(_seed_list[2]).uniform(0, 4, _kernel.beta.numel())).view_as(_kernel.beta)
        _x, _ = random_data(0, _n_choices, _n_data, _seed_list[3])
        _kernel_eval = _kernel(_x, _x, diag=False).evaluate()
        _eigval, _ = torch.symeig(_kernel_eval)
        _eigval_min = torch.min(_eigval).item()
        _eigval_max = torch.max(_eigval).item()
        print('KerVal : %+.4f ~ %.4f / Eigval : %+.4f ~ %.4f Ratio(%5.2f%%)'
              % (torch.min(_kernel_eval).item(), torch.max(_kernel_eval).item(),
                 _eigval_min, _eigval_max, abs(_eigval_min / _eigval_max * 100)))
