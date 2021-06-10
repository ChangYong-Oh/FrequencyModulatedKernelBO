import random

import numpy as np
import torch

import pyro

from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.util import optional


class Slice(MCMCKernel):
    """
    potential_fn: Python callable calculating potential energy (log likelihood) with input
        is a dict of real support parameters.


    :param grouping: in ['full', 'groupwise', 'elementwise']
        a dict of parameters is considered
        'full' : not distinguishing parameters coming from different keys (type),
                a list of one unit-norm vector is generated
        'groupwise' : for each key(type), random unit vector is generated.
                    a list of unit-norm vectors where nonzero values appear within the parameters of a key
        'elementwise' : a list of one-hot vectors
    :param shuffle_type: in ['full', 'both', 'groupwise', 'elementwise', None]
        None : no shuffle
        'full' : no effect if grouping in ['full', 'groupwise'], completely shuffle elementwise grouping
        'groupwise' : no effect if grouping == 'full', shuffle keywise
        'elementwise' : no effect if grouping in ['full', 'groupwise'], shuffle parameters with the same key
        'both' : no effect if grouping in ['full', 'groupwise'], apply groupwise and elementwise simultaneously
    """
    def __init__(self,
                 model,
                 width: float = 1,
                 max_steps_out: int = 10,
                 grouping='elementwise',
                 shuffle_type=None,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False):
        self.model = model
        self.transforms = transforms
        assert grouping in ['full', 'groupwise', 'elementwise']
        assert shuffle_type in ['full', 'both', 'groupwise', 'elementwise', None]
        self._width = width
        self._max_steps_out = max_steps_out
        self._grouping = grouping
        self._shuffle_type = shuffle_type
        self._max_plate_nesting = max_plate_nesting
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings
        self._reset()

        super(Slice, self).__init__()

    def _reset(self):
        self._t = 0
        self._prototype_trace = None
        self._initial_params = None
        self._z_last = None
        self._warmup_steps = None

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        if self._initial_params is None:
            self._initial_params = init_params
        self._params_name = list(self._initial_params.keys())
        self._params_size = {key: elm.size() for key, elm in self._initial_params.items()}
        self._params_numel = {key: elm.numel() for key, elm in self._initial_params.items()}
        self._prototype_trace = trace

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        self._initialize_model_properties(args, kwargs)
        potential_energy = self.potential_fn(self.initial_params)
        self._cache(self.initial_params, potential_energy)

    def cleanup(self):
        self._reset()

    def _cache(self, z, potential_energy):
        self._z_last = z
        self._potential_energy_last = potential_energy

    def clear_cache(self):
        self._z_last = None
        self._potential_energy_last = None

    def _fetch_from_cache(self):
        return self._z_last, self._potential_energy_last

    @staticmethod
    def _param_addv(param_origin, param_direction, length: float):
        """
            param_origin : param, a dict of real support parameters
            param_direction : a dict with the keys same as z, sum of square of all values should be 1
        """
        assert param_origin.keys() == param_direction.keys()
        return {key: param_origin[key] + length * param_direction[key].reshape_as(param_origin[key]) for key in param_origin.keys()}

    def _random_direction_group(self, param_name):
        random_param_direction = dict()
        for p_name in self._params_name:
            p_numel = self._params_numel[p_name]
            if p_name == param_name:
                direction_vec = torch.randn(p_numel) if p_numel > 1 else torch.ones(p_numel)
                direction_vec /= torch.sum(direction_vec ** 2) ** 0.5
            else:
                direction_vec = torch.zeros(p_numel)
            random_param_direction[p_name] = direction_vec.reshape(self._params_size[p_name])
        return random_param_direction

    def _elementwise_direction_group(self, param_name, i):
        assert 0 <= i < self._params_numel[param_name]
        random_param_direction = dict()
        for p_name in self._params_name:
            p_numel = self._params_numel[p_name]
            direction_vec = torch.zeros(p_numel)
            if p_name == param_name:
                direction_vec[i] = 1
            random_param_direction[p_name] = direction_vec.reshape(self._params_size[p_name])
        return random_param_direction

    def _random_direction_list(self):
        direction_list = []
        if self._grouping == 'full':
            random_vec = torch.randn(np.sum(list(self._params_numel.values())))
            random_vec /= torch.sum(random_vec ** 2) ** 0.5
            cnt = 0
            random_param_direction = dict()
            for p_name in self._params_name:
                p_numel = self._params_numel[p_name]
                random_param_direction[p_name] = random_vec[cnt:cnt+p_numel].reshape(self._params_size[p_name])
                cnt += p_numel
            direction_list.append(random_param_direction)
        elif self._grouping == 'groupwise':
            p_name_list = self._params_name
            if self._shuffle_type == 'groupwise':
                random.shuffle(p_name_list)
            for p_name in p_name_list:
                direction_list.append(self._random_direction_group(p_name))
        elif self._grouping == 'elementwise':
            p_name_list = self._params_name
            if self._shuffle_type in ['groupwise', 'both']:
                random.shuffle(p_name_list)
            for p_name in p_name_list:
                p_ind_list = list(range(self._params_numel[p_name]))
                if self._shuffle_type in ['elementwise', 'both']:
                    random.shuffle(p_ind_list)
                for n in p_ind_list:
                    direction_list.append(self._elementwise_direction_group(p_name, n))
            if self._shuffle_type == 'full':
                random.shuffle(direction_list)
        return direction_list

    def _directional_slice_sampling(self, z0, direction, potential_fn, width: float, max_steps_out: int):
        """
        z : param, a dict of real support parameters
        direction : a dict with the keys same as z,
        """
        lower = -width * torch.rand(1).item()
        upper = lower + width
        llh0 = potential_fn(z0)
        slice_h = torch.log(torch.rand(1)).item() + llh0

        steps_out = 0
        logp_lower = potential_fn(self._param_addv(z0, direction, lower))
        logp_upper = potential_fn(self._param_addv(z0, direction, upper))

        while (logp_lower > slice_h or logp_upper > slice_h) and (steps_out < max_steps_out):
            if torch.rand(1).item() < 0.5:
                lower -= (upper - lower)
                logp_lower = potential_fn(self._param_addv(z0, direction, lower))
            else:
                upper += (upper - lower)
                logp_upper = potential_fn(self._param_addv(z0, direction, upper))
            steps_out += 1

        # Shrinkage
        start_upper = upper
        start_lower = lower
        n_steps_in = 0
        while not np.isclose(lower, upper):
            step_size = (upper - lower) * torch.rand(1).item() + lower
            z1 = self._param_addv(z0, direction, step_size)
            llh1 = potential_fn(z1)
            if llh1 > slice_h and self._accept(z0, direction, step_size, potential_fn, slice_h, width, start_lower, start_upper):
                return z1
            else:
                if step_size < 0:
                    lower = step_size
                else:
                    upper = step_size
            n_steps_in += 1
        # raise RuntimeError('Shrinkage collapsed to a degenerated interval(point)')
        return z0  # just returning original value

    def _accept(self, z0, direction, step_size, potential_fn, slice_h, width, lower, upper):
        acceptance = False
        logp_upper = potential_fn(self._param_addv(z0, direction, upper))
        logp_lower = potential_fn(self._param_addv(z0, direction, lower))
        while upper - lower > 1.1 * width:
            mid = (lower + upper) / 2.0
            if (0 < mid <= step_size) or (0 >= mid > step_size):
                acceptance = True
            if step_size < mid:
                upper = mid
                logp_upper = potential_fn(self._param_addv(z0, direction, upper))
            else:
                lower = mid
                logp_lower = potential_fn(self._param_addv(z0, direction, lower))
            if acceptance and slice_h >= logp_lower and slice_h >= logp_upper:
                return False
        return True

    def sample(self, params):
        z, potential_energy = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            potential_energy = self.potential_fn(z)
            self._cache(z, potential_energy)
        # return early if no sample sites
        elif len(z) == 0:
            self._t += 1
            return params

        # TODO : check what below with statement does
        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            for direction in self._random_direction_list():
                z = self._directional_slice_sampling(z0=z, direction=direction, potential_fn=self.potential_fn,
                                                     width=self._width, max_steps_out=self._max_steps_out)

        return z.copy()
