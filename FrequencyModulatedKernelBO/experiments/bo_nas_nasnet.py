import os
import argparse
from datetime import datetime

from typing import Tuple, List

import torch
from torch import Tensor

from FrequencyModulatedKernelBO.experiments.data.blackbox_evaluator import BlackboxEvaluator
from FrequencyModulatedKernelBO.experiments.gp_models import KERNEL_TYPE_LIST
from FrequencyModulatedKernelBO.experiments.utils import save_bo_exp
from FrequencyModulatedKernelBO.experiments.config import exp_dir_root
from FrequencyModulatedKernelBO.experiments.data.exp_nas_nasnet import NASNetEvaluator, EXP_DATA
from FrequencyModulatedKernelBO.experiments.bo import print_bo_exp, init_bo, continue_bo


NASNET_EXP_DIR = exp_dir_root() + '_NASNET'


def init_evaluator_nasnet(exp_id: int):
    minimize = True
    assert exp_id in range(5)  # for the same exp, 5 different random initial data

    blackbox_evaluator = NASNetEvaluator(data=EXP_DATA, init_data_seed=exp_id, eval_init=True)

    initial_data: Tuple[Tensor, Tensor] = blackbox_evaluator.initial_data
    eval_x: Tensor = initial_data[0]
    eval_y: Tensor = initial_data[1]
    n_eval: int = eval_x.size(0)
    time_BO: Tensor = torch.ones(n_eval)
    time_eval: Tensor = torch.ones(n_eval)
    wallclock_time: List[datetime] = [datetime.now() for _ in range(n_eval)]

    save_eval_file = os.path.join(NASNET_EXP_DIR, 'NASNet_%s[E%d]_%s_[R%d]' %
                                  (blackbox_evaluator.data, eval_y.numel(),
                                   datetime.now().strftime('%y%m%d%H'), exp_id))
    save_bo_exp(save_file=save_eval_file, gp_model=None, acq_type=None, blackbox_evaluator=blackbox_evaluator,
                save_data_dict={'eval_x': eval_x, 'eval_y': eval_y,
                                'time_BO': time_BO, 'time_eval': time_eval, 'wallclock_time': wallclock_time,
                                'minimize': minimize},
                print_data_dict={'print_func': print_bo_exp,
                                 'gp_model_info_str': '',
                                 'blackbox_evaluator_info_str': blackbox_evaluator.info_str,
                                 'wallclock_time': wallclock_time, 'eval_x': eval_x, 'eval_y': eval_y,
                                 'minimize': minimize})
    return save_eval_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for architecture search [NASNet]')
    parser.add_argument('--exp_id', dest='exp_id', type=int, default=None, help='[0,1,2,3,4]')
    parser.add_argument('--model_type', dest='model_type', type=str, default=None, help=str(KERNEL_TYPE_LIST))
    parser.add_argument('--acq_type', dest='acq_type', type=str, default=None, help='ei, est')
    parser.add_argument('--save_path', dest='save_path', type=str, default=None)
    parser.add_argument('--job_type', dest='job_type', type=str, default=None, help='[suggest, evaluate, both]')
    parser.add_argument('--max_eval', dest='max_eval', type=int, default=None)

    args = parser.parse_args()
    if args.exp_id is not None:
        assert args.model_type is None and args.save_path is None and args.job_type is None and args.max_eval is None
        save_eval_file = init_evaluator_nasnet(args.exp_id)
        print('NASNet with initial data using random seed %d has been stored in the following file.')
        print(save_eval_file)
    else:
        if os.path.isfile(args.save_path):
            assert args.model_type is not None and args.job_type is None and args.max_eval is None
            save_dir = init_bo(save_file=args.save_path, model_type=args.model_type, acq_type=args.acq_type,
                               use_log_y=True, sampler_type=None, exp_dir=NASNET_EXP_DIR)#None: optimize
            print('The experiment can be continued using following directory.')
            print(save_dir)
        elif os.path.isdir(args.save_path):
            assert args.job_type.lower() in ['suggest', 'evaluate', 'both']
            do_suggest = True if args.job_type.lower() in ['suggest', 'both'] else False
            do_evaluate = True if args.job_type.lower() in ['evaluate', 'both'] else False
            continue_bo(save_dir=args.save_path, max_eval=args.max_eval, do_suggest=do_suggest, do_evaluate=do_evaluate)
        else:
            raise NotImplementedError

