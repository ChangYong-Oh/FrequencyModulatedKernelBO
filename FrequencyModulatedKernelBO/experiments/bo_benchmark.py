import os
import argparse
from datetime import datetime

from typing import Tuple, List

import torch
from torch import Tensor

from FrequencyModulatedKernelBO.experiments.gp_models import KERNEL_TYPE_LIST
from FrequencyModulatedKernelBO.experiments.utils import save_bo_exp
from FrequencyModulatedKernelBO.experiments.config import exp_dir_root, FILENAME_ZFILL_SIZE
from FrequencyModulatedKernelBO.experiments.bo import print_bo_exp, init_bo, continue_bo
from FrequencyModulatedKernelBO.experiments.data import Func2CEvaluator, Func3CEvaluator, Ackley5CEvaluator, \
    SVMBostonEvaluator, XGBFashionMNISTEvaluator


BENCHMARK_EXP_TYPE = ['Func2C', 'Func3C', 'Ackley5C', 'SVMBoston', 'XGBFashionMNIST']


def init_evaluator_benchmark(exp_type: str, exp_id: int, minimize: bool = True):
    assert exp_type in BENCHMARK_EXP_TYPE
    assert exp_id in range(5)  # for the same exp, 5 different random initial data

    if exp_type == 'Func2C':
        blackbox_evaluator = Func2CEvaluator(exp_id, eval_init=True)
    elif exp_type == 'Func3C':
        blackbox_evaluator = Func3CEvaluator(exp_id, eval_init=True)
    elif exp_type == 'Ackley5C':
        blackbox_evaluator = Ackley5CEvaluator(exp_id, eval_init=True)
    elif exp_type == 'SVMBoston':
        blackbox_evaluator = SVMBostonEvaluator(exp_id, eval_init=True)
    elif exp_type == 'XGBFashionMNIST':
        blackbox_evaluator = XGBFashionMNISTEvaluator(exp_id, eval_init=True)
    else:
        raise NotImplementedError

    initial_data: Tuple[Tensor, Tensor] = blackbox_evaluator.initial_data
    eval_x: Tensor = initial_data[0]
    eval_y: Tensor = initial_data[1]
    n_eval: int = eval_x.size(0)
    time_BO: Tensor = torch.ones(n_eval)
    time_eval: Tensor = torch.ones(n_eval)
    wallclock_time: List[datetime] = [datetime.now() for _ in range(n_eval)]

    save_eval_file = os.path.join(exp_dir_root(),
                                  '_'.join([blackbox_evaluator.info_str, '[R%d]' % exp_id]))
    save_bo_exp(save_file=save_eval_file, gp_model=None, acq_type=None, blackbox_evaluator=blackbox_evaluator,
                save_data_dict={'eval_x': eval_x, 'eval_y': eval_y,
                                'time_BO': time_BO, 'time_eval': time_eval, 'wallclock_time': wallclock_time},
                print_data_dict={'print_func': print_bo_exp,
                                 'gp_model_info_str': '',
                                 'blackbox_evaluator_info_str': blackbox_evaluator.info_str,
                                 'wallclock_time': wallclock_time, 'eval_x': eval_x, 'eval_y': eval_y,
                                 'minimize': minimize})
    return save_eval_file


def start_bo_benchmark(exp_type: str, exp_id: int, model_type: str, max_eval: int):
    save_eval_file = init_evaluator_benchmark(exp_type=exp_type, exp_id=exp_id) + '.pkl'
    save_dir = init_bo(save_file=save_eval_file, model_type=model_type, sampler_type=None)  # to use optimize
    continue_bo(save_dir=save_dir, max_eval=max_eval, do_suggest=True, do_evaluate=True)
    return save_dir


def continue_bo_benchmark(save_dir, max_eval: int):
    continue_bo(save_dir=save_dir, max_eval=max_eval, do_suggest=True, do_evaluate=True)
    return save_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimization for NK model')
    parser.add_argument('--exp_type', dest='exp_type', type=str, default=None, help=str(BENCHMARK_EXP_TYPE))
    parser.add_argument('--exp_id', dest='exp_id', type=int, default=None, help='[0,1,2,3,4]')
    parser.add_argument('--model_type', dest='model_type', type=str, default=None, help=str(KERNEL_TYPE_LIST))
    parser.add_argument('--acq_type', dest='acq_type', type=str, default=None, help='ei, est')
    parser.add_argument('--save_path', dest='save_path', type=str, default=None)
    parser.add_argument('--max_eval', dest='max_eval', type=int, default=None)

    args = parser.parse_args()
    if args.exp_id is not None:
        assert args.model_type is None and args.save_path is None and args.max_eval is None
        init_evaluator_benchmark(exp_type=args.exp_type, exp_id=args.exp_id)
    else:
        if args.model_type is not None:
            assert os.path.isfile(args.save_path) and args.max_eval is None
            init_bo(save_file=args.save_path, model_type=args.model_type, acq_type=args.acq_type,
                    use_log_y=False, sampler_type=None)  # None means optimize
        else:
            continue_bo(save_dir=args.save_path, max_eval=args.max_eval, do_suggest=True, do_evaluate=True)
