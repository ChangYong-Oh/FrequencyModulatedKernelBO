import warnings

import torch
import torch.multiprocessing as mp

import pyro
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.util import diagnostics, initialize_model, print_summary
from pyro.infer.mcmc.api import _UnarySampler, _MultiSampler
import pyro.poutine as poutine

from FrequencyModulatedKernelBO.pyro_cbo.infer.mcmc.slice import Slice


class MCMC(object):
    """
    Wrapper class for Markov Chain Monte Carlo algorithms. Specific MCMC algorithms
    are TraceKernel instances and need to be supplied as a ``kernel`` argument
    to the constructor.
    .. note:: The case of `num_chains > 1` uses python multiprocessing to
        run parallel chains in multiple processes. This goes with the usual
        caveats around multiprocessing in python, e.g. the model used to
        initialize the ``kernel`` must be serializable via `pickle`, and the
        performance / constraints will be platform dependent (e.g. only
        the "spawn" context is available in Windows). This has also not
        been extensively tested on the Windows platform.
    :param kernel: An instance of the ``TraceKernel`` class, which when
        given an execution trace returns another sample trace from the target
        (posterior) distribution.
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int warmup_steps: Number of warmup iterations. The samples generated
        during the warmup phase are discarded. If not provided, default is
        half of `num_samples`.
    :param int num_chains: Number of MCMC chains to run in parallel. Depending on
        whether `num_chains` is 1 or more than 1, this class internally dispatches
        to either `_UnarySampler` or `_MultiSampler`.
    :param dict initial_params: dict containing initial tensors in unconstrained
        space to initiate the markov chain. The leading dimension's size must match
        that of `num_chains`. If not specified, parameter values will be sampled from
        the prior.
    :param hook_fn: Python callable that takes in `(kernel, samples, stage, i)`
        as arguments. stage is either `sample` or `warmup` and i refers to the
        i'th sample for the given stage. This can be used to implement additional
        logging, or more generally, run arbitrary code per generated sample.
    :param str mp_context: Multiprocessing context to use when `num_chains > 1`.
        Only applicable for Python 3.5 and above. Use `mp_context="spawn"` for
        CUDA.
    :param bool disable_progbar: Disable progress bar and diagnostics update.
    :param bool disable_validation: Disables distribution validation check. This is
        disabled by default, since divergent transitions will lead to exceptions.
        Switch to `True` for debugging purposes.
    :param dict transforms: dictionary that specifies a transform for a sample site
        with constrained support to unconstrained space.
    """
    def __init__(self, kernel, num_samples, warmup_steps=None, initial_params=None,
                 num_chains=1, hook_fn=None, mp_context=None, disable_progbar=False,
                 disable_validation=True, transforms=None):
        self.warmup_steps = num_samples if warmup_steps is None else warmup_steps  # Stan
        self.num_samples = num_samples
        self.kernel = kernel
        self.transforms = transforms
        self.disable_validation = disable_validation
        self._samples = None
        self._args = None
        self._kwargs = None
        if isinstance(self.kernel, (HMC, NUTS)) and self.kernel.potential_fn is not None:
            if initial_params is None:
                raise ValueError("Must provide valid initial parameters to begin sampling"
                                 " when using `potential_fn` in HMC/NUTS kernel.")
        parallel = False
        if num_chains > 1:
            # check that initial_params is different for each chain
            if initial_params:
                for v in initial_params.values():
                    if v.shape[0] != num_chains:
                        raise ValueError("The leading dimension of tensors in `initial_params` "
                                         "must match the number of chains.")
                # FIXME: probably we want to use "spawn" method by default to avoid the error
                # CUDA initialization error https://github.com/pytorch/pytorch/issues/2517
                # even that we run MCMC in CPU.
                if mp_context is None:
                    # change multiprocessing context to 'spawn' for CUDA tensors.
                    if list(initial_params.values())[0].is_cuda:
                        mp_context = "spawn"

            # verify num_chains is compatible with available CPU.
            available_cpu = max(mp.cpu_count() - 1, 1)  # reserving 1 for the main process.
            if num_chains <= available_cpu:
                parallel = True
            else:
                warnings.warn("num_chains={} is more than available_cpu={}. "
                              "Chains will be drawn sequentially."
                              .format(num_chains, available_cpu))
        else:
            if initial_params:
                initial_params = {k: v.unsqueeze(0) for k, v in initial_params.items()}

        self.num_chains = num_chains
        self._diagnostics = [None] * num_chains

        if parallel:
            self.sampler = _MultiSampler(kernel, num_samples, self.warmup_steps, num_chains, mp_context,
                                         disable_progbar, initial_params=initial_params, hook=hook_fn)
        else:
            self.sampler = _UnarySampler(kernel, num_samples, self.warmup_steps, num_chains, disable_progbar,
                                         initial_params=initial_params, hook=hook_fn)

    @poutine.block
    def run(self, *args, **kwargs):
        self._args, self._kwargs = args, kwargs
        num_samples = [0] * self.num_chains
        z_flat_acc = [[] for _ in range(self.num_chains)]
        with pyro.validation_enabled(not self.disable_validation):
            for x, chain_id in self.sampler.run(*args, **kwargs):
                if num_samples[chain_id] == 0:
                    num_samples[chain_id] += 1
                    z_structure = x
                elif num_samples[chain_id] == self.num_samples + 1:
                    self._diagnostics[chain_id] = x
                else:
                    num_samples[chain_id] += 1
                    if self.num_chains > 1:
                        x_cloned = x.clone()
                        del x
                    else:
                        x_cloned = x
                    z_flat_acc[chain_id].append(x_cloned)

        z_flat_acc = torch.stack([torch.stack(l) for l in z_flat_acc])

        # unpack latent
        pos = 0
        z_acc = z_structure.copy()
        for k in sorted(z_structure):
            shape = z_structure[k]
            next_pos = pos + shape.numel()
            z_acc[k] = z_flat_acc[:, :, pos:next_pos].reshape(
                (self.num_chains, self.num_samples) + shape)
            pos = next_pos
        assert pos == z_flat_acc.shape[-1]

        # If transforms is not explicitly provided, infer automatically using
        # model args, kwargs.
        if self.transforms is None and isinstance(self.kernel, (HMC, NUTS, Slice)):# only modification from the orginal one
            if self.kernel.transforms is not None:
                self.transforms = self.kernel.transforms
            elif self.kernel.model:
                _, _, self.transforms, _ = initialize_model(self.kernel.model,
                                                            model_args=args,
                                                            model_kwargs=kwargs)
            else:
                self.transforms = {}

        # transform samples back to constrained space
        for name, transform in self.transforms.items():
            z_acc[name] = transform.inv(z_acc[name])
        self._samples = z_acc

        # terminate the sampler (shut down worker processes)
        self.sampler.terminate(True)

    def get_samples(self, num_samples=None, group_by_chain=False):
        """
        Get samples from the MCMC run, potentially resampling with replacement.
        :param int num_samples: Number of samples to return. If `None`, all the samples
            from an MCMC chain are returned in their original ordering.
        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            all samples will have num_chains as the size of their leading dimension.
        :return: dictionary of samples keyed by site name.
        """
        samples = self._samples
        if num_samples is None:
            # reshape to collapse chain dim when group_by_chain=False
            if not group_by_chain:
                samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
        else:
            if not samples:
                raise ValueError("No samples found from MCMC run.")
            if group_by_chain:
                batch_dim = 1
            else:
                samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
                batch_dim = 0
            sample_tensor = list(samples.values())[0]
            batch_size, device = sample_tensor.shape[batch_dim], sample_tensor.device
            idxs = torch.randint(0, batch_size, size=(num_samples,), device=device)
            samples = {k: v.index_select(batch_dim, idxs) for k, v in samples.items()}
        return samples

    def diagnostics(self):
        """
        Gets some diagnostics statistics such as effective sample size, split
        Gelman-Rubin, or divergent transitions from the sampler.
        """
        diag = diagnostics(self._samples)
        for diag_name in self._diagnostics[0]:
            diag[diag_name] = {'chain {}'.format(i): self._diagnostics[i][diag_name]
                               for i in range(self.num_chains)}
        return diag

    def summary(self, prob=0.9):
        """
        Prints a summary table displaying diagnostics of samples obtained from
        posterior. The diagnostics displayed are mean, standard deviation, median,
        the 90% Credibility Interval, :func:`~pyro.ops.stats.effective_sample_size`,
        :func:`~pyro.ops.stats.split_gelman_rubin`.
        :param float prob: the probability mass of samples within the credibility interval.
        """
        print_summary(self._samples, prob=prob)
        if 'divergences' in self._diagnostics[0]:
            print("Number of divergences: {}".format(
                sum([len(self._diagnostics[i]['divergences']) for i in range(self.num_chains)])))