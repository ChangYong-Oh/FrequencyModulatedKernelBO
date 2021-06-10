import os
import math
import argparse
import subprocess
from datetime import datetime

from typing import Tuple, List, Dict

import numpy as np

import torch
from torch import Tensor

from FrequencyModulatedKernelBO.experiments.utils import load_bo_exp
from FrequencyModulatedKernelBO.experiments.config import exp_dir_root, FILENAME_ZFILL_SIZE
from FrequencyModulatedKernelBO.experiments.data.exp_nas_nasnet import NASNetEvaluator, N_STATES
from FrequencyModulatedKernelBO.experiments.utils import last_file_in_directory


POPULATION_SIZE = 50
SAMPLE_SIZE = 15
MUTATION_STD = 0.01


def print_re_exp(nasnet_evaluator_info_str: str,
                 wallclock_time: List[datetime], eval_y: Tensor, best_eval: Tensor, best_eval_ind: Tensor) -> str:
    n_eval = eval_y.size(0)
    n_digit = int(1 + math.ceil(math.log10(n_eval)))
    format_str = "%s %s-th eval: %+.6E / best: %+.6E at %s"
    print_str = str(subprocess.Popen(['git', 'log', '-1'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0], 'utf-8')
    for _ in range(3):
        print_str += '=' * 80 + '\n'
    for i in range(n_eval):
        print_str += format_str % (
            wallclock_time[i].strftime('%H:%M:%S'),
            str(i + 1).zfill(n_digit), eval_y[i].item(), best_eval[i].item(),
            str(best_eval_ind[i].item()).zfill(n_digit))
        print_str += '\n'
    print_str += 'Blackbox function : %s\n' % nasnet_evaluator_info_str
    print_str += 'Model : Regularized Evolution(P:%d/S:%d)\n' % (POPULATION_SIZE, SAMPLE_SIZE)
    return print_str


def generate_re_exp_tag(nasnet_evaluator: NASNetEvaluator, exp_id: int) -> str:
    time_tag = datetime.now().strftime('%m%d-%H%M%S-%f')
    exp_id = '_'.join([nasnet_evaluator.info_str, 'RegularizedEvolution', '[R%d]' % exp_id, time_tag])
    return exp_id


def save_re_exp(save_file: str, nasnet_evaluator: NASNetEvaluator,
                save_data_dict: Dict, print_data_dict: Dict, display=True):
    save_dict = dict()
    save_dict['nasnet_evaluator'] = nasnet_evaluator.state_dict()
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


def load_re_exp(save_file: str) -> Tuple[NASNetEvaluator, Dict]:
    exp_info = torch.load(save_file)
    exp_info_nasnet_evaluator = exp_info['nasnet_evaluator']
    nasnet_evaluator_name = exp_info_nasnet_evaluator['name']
    del exp_info_nasnet_evaluator['name']
    nasnet_evaluator = globals()[nasnet_evaluator_name](**exp_info_nasnet_evaluator)

    return nasnet_evaluator, exp_info['data']


def random_child(nasnet_evaluator: NASNetEvaluator) -> torch.Tensor:
    x = torch.zeros(1, nasnet_evaluator.n_continuous_vars + len(nasnet_evaluator.list_of_categories))
    for i in range(x.numel()):
        if i < nasnet_evaluator.n_continuous_vars:
            lower, upper = nasnet_evaluator.optimized_features[i]
            x[0, i] = torch.rand(1) * (upper - lower)  + lower
        else:
            x[0, i] = torch.randint(low=0,
                                    high=nasnet_evaluator.list_of_categories[i - nasnet_evaluator.n_continuous_vars],
                                    size=(1,))
    return x


def mutated_child(nasnet_evaluator: NASNetEvaluator, parent: Tensor) -> torch.Tensor:
    if parent.dim() == 2:
        parent = parent.view(-1)

    # Minimization
    new_x = parent.clone()

    n_continuous_vars = nasnet_evaluator.n_continuous_vars

    continuous_mutation_ind = np.random.randint(0, n_continuous_vars)
    lower, upper = nasnet_evaluator.optimized_features[continuous_mutation_ind]
    new_x[continuous_mutation_ind] = (parent[continuous_mutation_ind] +
                                      torch.randn((1,)) * (upper - lower) * MUTATION_STD).clamp(min=lower, max=upper)
    if torch.randn((1,)) > 0:#mutate state input
        discrete_mutation_ind = np.random.randint(0, N_STATES)
        candidate_encoding = nasnet_evaluator.category_encoding[2 * N_STATES + discrete_mutation_ind]
        parent_encoding = candidate_encoding[parent[n_continuous_vars + 2 * N_STATES + discrete_mutation_ind].long()]
        if isinstance(parent_encoding, int):
            candidate_mutation = [c for c in range(len(candidate_encoding))
                                  if c != parent[n_continuous_vars + 2 * N_STATES + discrete_mutation_ind]]
            new_x[n_continuous_vars + 2 * N_STATES + discrete_mutation_ind] = np.random.choice(candidate_mutation)
        elif isinstance(parent_encoding, tuple):
            while True:
                which_input = np.random.randint(0, 2)
                candidate_mutation = [c for c, cand in enumerate(candidate_encoding)
                                      if (cand[which_input] == parent_encoding[which_input]
                                          and cand[1 - which_input] != parent_encoding[1 - which_input])]
                if len(candidate_mutation) >0:
                    break
            new_x[n_continuous_vars + 2 * N_STATES + discrete_mutation_ind] = np.random.choice(candidate_mutation)
        else:
            raise NotImplementedError
    else:#mutate operation
        discrete_mutation_ind = torch.randint(low=0, high=2 * N_STATES, size=(1,)).item()
        candidate_encoding = nasnet_evaluator.category_encoding[discrete_mutation_ind]
        candidate_mutation = [c for c in range(len(candidate_encoding))
                              if c != parent[n_continuous_vars + discrete_mutation_ind]]
        new_x[n_continuous_vars + discrete_mutation_ind] = np.random.choice(candidate_mutation)

    return new_x.view(1, -1)


def evolve(nasnet_evaluator: NASNetEvaluator,
           eval_x: Tensor, eval_y: Tensor, population_x: Tensor, population_y: Tensor,
           time_BO: Tensor, time_eval: Tensor, wallclock_time: List[datetime],
           save_file_prefix: str):
    minimize = True
    assert eval_x.size(0) == time_BO.size(0) == eval_y.size(0) == time_eval.size(0) == len(wallclock_time)

    torch_argopt = torch.argmin if minimize else torch.argmax

    print('[%s]%3d-th suggestion began' % (datetime.now().strftime('%H:%M:%S'), eval_x.size(0) + 1))

    datetime_BO_begin = datetime.now()
    if eval_x.size(0) < POPULATION_SIZE:
        print('The current population size is %d, therefore a random child is added to it.' % population_x.size(0))
        new_x = random_child(nasnet_evaluator=nasnet_evaluator)
    else:
        print('A mutated child is added and the oldest one is discarded.')
        sample_ind = torch.randint(0, population_x.size(0), (SAMPLE_SIZE,)) # sample with replacement
        sampled_x, sampled_y = population_x[sample_ind], population_y[sample_ind]
        parent = sampled_x[torch_argopt(sampled_y)]
        new_x = mutated_child(nasnet_evaluator=nasnet_evaluator, parent=parent)
    datetime_BO_end = datetime.now()
    eval_x = torch.cat([eval_x, new_x])
    time_BO = torch.cat([time_BO, (datetime_BO_begin - datetime_BO_end).total_seconds() * torch.ones(1)])

    datatime_eval_begin = datetime.now()
    new_y = nasnet_evaluator(eval_x[-1])
    datetime_eval_end = datetime.now()
    eval_y = torch.cat([eval_y, new_y.view(1, 1)], dim=0)

    population_x = torch.cat([population_x, new_x], dim=0)[-POPULATION_SIZE:]
    population_y = torch.cat([population_y, new_y.view(1, 1)], dim=0)[-POPULATION_SIZE:]

    time_eval = torch.cat([time_eval, (datatime_eval_begin - datetime_eval_end).total_seconds() * torch.ones(1)])
    wallclock_time.append(datetime_eval_end)

    save_data_dict = {'eval_x': eval_x, 'eval_y': eval_y,
                      'population_x': population_x, 'population_y': population_y,
                      'time_BO': time_BO, 'time_eval': time_eval, 'wallclock_time': wallclock_time,
                      'minimize': minimize}
    save_file = '_'.join([save_file_prefix, str(eval_x.size(0)).zfill(FILENAME_ZFILL_SIZE)])
    save_re_exp(save_file=save_file, nasnet_evaluator=nasnet_evaluator,
                save_data_dict=save_data_dict,
                print_data_dict={'print_func': print_re_exp,
                                 'nasnet_evaluator_info_str': nasnet_evaluator.info_str,
                                 'wallclock_time': wallclock_time, 'eval_x': eval_x, 'eval_y': eval_y,
                                 'minimize': minimize},
                display=False)


def init_re_nasnet(save_file: str) -> str:
    minimize = True
    exp_id = int(os.path.basename(os.path.normpath(save_file)).split('-')[0][-2:-1])
    _, _, nasnet_evaluator, BO_data = load_bo_exp(save_file)
    eval_x = BO_data['eval_x']
    eval_y = BO_data['eval_y']
    population_x = eval_x.clone()
    population_y = eval_y.clone()
    time_BO = BO_data['time_BO']
    time_eval = BO_data['time_eval']
    wallclock_time = BO_data['wallclock_time']

    exp_tag = generate_re_exp_tag(nasnet_evaluator=nasnet_evaluator, exp_id=exp_id)
    save_dir = os.path.join(exp_dir_root(), exp_tag)
    os.makedirs(save_dir)
    save_file_prefix = os.path.join(save_dir, exp_tag)

    save_data_dict = {'eval_x': eval_x, 'eval_y': eval_y,
                      'population_x': population_x, 'population_y': population_y,
                      'time_BO': time_BO, 'time_eval': time_eval, 'wallclock_time': wallclock_time,
                      'minimize': minimize}
    save_file = '_'.join([save_file_prefix, str(eval_x.size(0)).zfill(FILENAME_ZFILL_SIZE)])
    save_re_exp(save_file=save_file, nasnet_evaluator=nasnet_evaluator,
                save_data_dict=save_data_dict,
                print_data_dict={'print_func': print_re_exp,
                                 'nasnet_evaluator_info_str': nasnet_evaluator.info_str,
                                 'wallclock_time': wallclock_time, 'eval_x': eval_x, 'eval_y': eval_y,
                                 'minimize': minimize})
    return save_dir


def run_re(save_dir: str, max_eval: int):
    while True:
        last_file, last_file_prefix = last_file_in_directory(save_dir)
        nasnet_evaluator, RE_data = load_re_exp(last_file)
        if RE_data['eval_y'].size(0) >= max_eval:
            break
        evolve(nasnet_evaluator=nasnet_evaluator, save_file_prefix=last_file_prefix, **RE_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regularized Evolution for architecture search [NASNet]')
    parser.add_argument('--save_path', dest='save_path', type=str, default=None)
    parser.add_argument('--max_eval', dest='max_eval', type=int, default=None)

    args = parser.parse_args()
    if os.path.isfile(args.save_path):
        save_dir = init_re_nasnet(args.save_path)
    elif os.path.isdir(args.save_path):
        save_dir = args.save_path
    else:
        raise ValueError('save_path argument should be a proper file or directory path.')
    run_re(save_dir=save_dir, max_eval=args.max_eval)