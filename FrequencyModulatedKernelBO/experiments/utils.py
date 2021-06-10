from typing import Tuple, Dict, Optional

import os
import datetime

import torch

from FrequencyModulatedKernelBO.experiments.gp_models import MixedVariableGPBase, MixedVariableGPMCMC, MixedVariableGPOptimize
from FrequencyModulatedKernelBO.experiments.data.blackbox_evaluator import BlackboxEvaluator
from FrequencyModulatedKernelBO.experiments.config import FILENAME_ZFILL_SIZE
from FrequencyModulatedKernelBO.experiments.data import \
    Func2CEvaluator, Func3CEvaluator, Ackley5CEvaluator, \
    SVMBostonEvaluator, XGBFashionMNISTEvaluator, NASNetEvaluator


def last_file_in_directory(save_dir: str):
    last_eval_num = max([int(os.path.splitext(elm)[0].split('_')[-1]) for elm in os.listdir(save_dir)
                         if os.path.splitext(elm)[1] == '.pkl'])
    last_file = [elm for elm in os.listdir(save_dir)
                 if str(last_eval_num).zfill(FILENAME_ZFILL_SIZE) + '.pkl' in elm][0]
    file_prefix = '_'.join(os.path.splitext(last_file)[0].split('_')[:-1])
    save_file = os.path.join(save_dir, last_file)
    save_file_prefix = os.path.join(save_dir, file_prefix)
    return save_file, save_file_prefix


def generate_exp_tag(gp_model: MixedVariableGPBase, acq_type: str, blackbox_evaluator: BlackboxEvaluator, exp_id: int) -> str:
    time_tag = datetime.datetime.now().strftime('%m%d-%H%M%S-%f')
    exp_id = '_'.join([blackbox_evaluator.info_str, gp_model.info_str, acq_type, '[R%d]' % exp_id, time_tag])
    return exp_id


def save_bo_exp(save_file: str, gp_model: Optional[MixedVariableGPBase], acq_type: Optional[str],
                blackbox_evaluator: BlackboxEvaluator, save_data_dict: Dict, print_data_dict: Dict, display=True):
    save_dict = dict()
    save_dict['gp_model'] = gp_model.state_dict() if gp_model is not None else None
    save_dict['acq_type'] = acq_type
    save_dict['blackbox_evaluator'] = blackbox_evaluator.state_dict()
    save_dict['data'] = save_data_dict
    torch.save(save_dict, save_file + '.pkl')
    print_kwargs = print_data_dict.copy()
    print_func = print_kwargs['print_func']
    del print_kwargs['print_func']
    print_str = print_func(**print_kwargs)
    with open(save_file + '.txt', 'wt') as f:
        f.write(print_str)
    if display:
        print(print_str)


def load_bo_exp(save_file: str) -> Tuple[Optional[MixedVariableGPBase], str, BlackboxEvaluator, Dict]:
    exp_info = torch.load(save_file)
    exp_info_gp_model = exp_info['gp_model']
    acq_type = exp_info['acq_type']
    if exp_info_gp_model is not None:
        if 'last_sample_original' in exp_info_gp_model:
            last_sample_unconstrained = exp_info_gp_model['last_sample_unconstrained']
            last_sample_original = exp_info_gp_model['last_sample_original']
            del exp_info_gp_model['last_sample_unconstrained']
            del exp_info_gp_model['last_sample_original']
            gp_model = MixedVariableGPMCMC(**exp_info['gp_model'])
            gp_model.last_sample_unconstrained = last_sample_unconstrained
            gp_model.last_sample_original = last_sample_original
        elif 'n_runs' in exp_info_gp_model:
            n_runs = exp_info_gp_model['n_runs']
            del exp_info_gp_model['n_runs']
            gp_model = MixedVariableGPOptimize(**exp_info['gp_model'])
            gp_model.n_runs = n_runs
    else:
        gp_model = None

    exp_info_blackbox_evaluator = exp_info['blackbox_evaluator']
    blackbox_evaluator_name = exp_info_blackbox_evaluator['name']
    del exp_info_blackbox_evaluator['name']
    blackbox_evaluator = globals()[blackbox_evaluator_name](**exp_info_blackbox_evaluator)

    return gp_model, acq_type, blackbox_evaluator, exp_info['data']

