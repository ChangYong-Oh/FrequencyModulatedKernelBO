import os
import argparse

from FrequencyModulatedKernelBO.experiments.config import exp_dir_root
from FrequencyModulatedKernelBO.experiments.gp_models import KERNEL_TYPE_LIST
from FrequencyModulatedKernelBO.experiments.bo_benchmark import BENCHMARK_EXP_TYPE, start_bo_benchmark, continue_bo_benchmark


EXP_TYPES = BENCHMARK_EXP_TYPE


if __name__ == '__main__':
    exp_type = 'exp_type'
    exp_id = 'exp_id'
    model_type = 'model_type'
    save_dir = 'save_dir'
    max_eval = 'max_eval'

    parser_ = argparse.ArgumentParser(
        description='Bayesian Optimization for Dynamically Evolving Objectives')
    parser_.add_argument('--' + exp_type, dest=exp_type, type=str, default=None, help=str(EXP_TYPES))
    parser_.add_argument('--' + exp_id, dest=exp_id, type=int, default=None, help='[0,1,2,3,4]')
    parser_.add_argument('--' + model_type, dest=model_type, type=str, default=None, help=str(KERNEL_TYPE_LIST))
    parser_.add_argument('--' + save_dir, dest=save_dir, type=str, default=None)
    parser_.add_argument('--' + max_eval, dest=max_eval, type=int, default=None)

    _kwargs = vars(parser_.parse_args())
    if _kwargs[save_dir] is not None:
        _kwargs[save_dir] = os.path.join(exp_dir_root(), _kwargs[save_dir])
    assert (_kwargs[exp_type] is None) == (_kwargs[exp_id] is None) == (_kwargs[model_type] is None)
    assert (_kwargs[exp_type] is None) != (_kwargs[save_dir] is None)
    if _kwargs[save_dir] is None:
        # 5 different initializations are tested
        assert _kwargs[exp_type] in EXP_TYPES
        assert _kwargs[exp_id] in range(5)
        assert _kwargs[model_type] in KERNEL_TYPE_LIST
    else:
        assert os.path.exists(_kwargs[save_dir])

    if _kwargs[save_dir] is None:
        del _kwargs[save_dir]
        assert _kwargs[exp_type] in EXP_TYPES
        if _kwargs[exp_type] in BENCHMARK_EXP_TYPE:
            _save_dir = start_bo_benchmark(**_kwargs)
        else:
            raise NotImplementedError
    else:
        _save_dir = os.path.split(_kwargs[save_dir])[1]
        if _save_dir == '':
            _save_dir = os.path.split(os.path.split(_kwargs[save_dir])[0])[1]
        if [elm in _save_dir for elm in BENCHMARK_EXP_TYPE].count(True) == 1:
            _save_dir = continue_bo_benchmark(save_dir=_kwargs[save_dir], max_eval=_kwargs[max_eval])
        else:
            raise NotImplementedError
    print(_save_dir)
