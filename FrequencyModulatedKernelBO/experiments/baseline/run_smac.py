from typing import Optional

import os
import numpy as np
import pickle
import argparse
from datetime import datetime

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace, Configuration
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.facade.func_facade import fmin_smac

import torch

from FrequencyModulatedKernelBO.experiments.data.blackbox_evaluator import BlackboxEvaluator
from FrequencyModulatedKernelBO.experiments.config import exp_dir_root
from FrequencyModulatedKernelBO.experiments.data import Func2CEvaluator, Func3CEvaluator, Ackley5CEvaluator, \
    SVMBostonEvaluator, XGBFashionMNISTEvaluator


def generate_exp_tag(blackbox_evaluator: BlackboxEvaluator, exp_id: int) -> str:
    time_tag = datetime.now().strftime('%m%d-%H%M%S-%f')
    exp_id = '_'.join([blackbox_evaluator.info_str, '[R%d]' % exp_id, 'SMAC', time_tag])
    return exp_id


def smac_wrapper(n_eval, exp_type, exp_id, save_dir: Optional[str] = None):
    assert exp_id in range(5)

    if exp_type == 'Func2C':
        blackbox_evaluator = Func2CEvaluator(exp_id, eval_init=True)
    elif exp_type == 'Func3C':
        blackbox_evaluator = Func3CEvaluator(exp_id, eval_init=True)
    elif exp_type == 'Ackley5C':
        blackbox_evaluator = Ackley5CEvaluator(exp_id, eval_init=True)
    elif exp_type == 'SVMBoston':
        blackbox_evaluator = SVMBostonEvaluator(exp_id, eval_init=True)
    elif exp_type == 'XGBFashionMNIST':
        blackbox_evaluator = XGBFashionMNISTEvaluator(exp_id, eval_init=False)
    else:
        raise NotImplementedError

    n_continuous = blackbox_evaluator.n_continuous
    n_ordinal = blackbox_evaluator.n_ordinal
    n_nominal = blackbox_evaluator.n_nominal
    n_variable = blackbox_evaluator.n_variable

    cs = ConfigurationSpace()
    for i, (lower, upper) in enumerate(blackbox_evaluator.list_of_contiuous):
        continuous_var = UniformFloatHyperparameter('x' + str(i + 1).zfill(2), lower=lower, upper=upper)
        cs.add_hyperparameter(continuous_var)
    for i, n_values in enumerate(blackbox_evaluator.list_of_ordinal):
        ordinal_var = UniformIntegerHyperparameter('x' + str(i + blackbox_evaluator.shift_ordinal + 1).zfill(2),
                                                   lower=0, upper=n_values - 1)
        cs.add_hyperparameter(ordinal_var)
    for i, n_values in enumerate(blackbox_evaluator.list_of_nominal):
        nominal_var = CategoricalHyperparameter('x' + str(i + blackbox_evaluator.shift_nominal + 1).zfill(2),
                                                [str(int(elm)) for elm in range(n_values)])
        cs.add_hyperparameter(nominal_var)

    if exp_type == 'XGBFashionMNIST':
        with open(os.path.join(exp_dir_root(), '%s_[R%d].pkl' % (exp_type, exp_id)), 'rb') as f:
            data = torch.load(f)['data']
        initial_data_x = data['eval_x']
        initial_data_y = data['eval_y']
    else:
        initial_data_x, initial_data_y = blackbox_evaluator.initial_data
    n_initial_data = initial_data_x.size(0)
    initial_data_x_smac_format = []
    for i in range(initial_data_x.size(0)):
        data_x_smac_format_dict = dict()
        for j in range(n_continuous):
            data_x_smac_format_dict['x' + str(j + 1).zfill(2)] = initial_data_x[i, j].item()
        for j in range(n_continuous, n_continuous + n_ordinal):
            data_x_smac_format_dict['x' + str(j + 1).zfill(2)] = int(initial_data_x[i, j].item())
        for j in range(n_continuous + n_ordinal, n_continuous + n_ordinal + n_nominal):
            data_x_smac_format_dict['x' + str(j + 1).zfill(2)] = str(int(initial_data_x[i, j].item()))
        initial_data_x_smac_format.append(Configuration(cs, data_x_smac_format_dict))

    def evaluate(x):
        x_tensor = torch.zeros(n_variable)
        for d in range(n_continuous):
            x_tensor[d] = x['x' + str(d + 1).zfill(2)]
        for d in range(n_continuous, n_continuous + n_ordinal):
            x_tensor[d] = x['x' + str(d + 1).zfill(2)]
        for d in range(n_continuous + n_ordinal, n_continuous + n_ordinal + n_nominal):
            x_tensor[d] = int(x['x' + str(d + 1).zfill(2)])
        return blackbox_evaluator(x_tensor.view(1, -1)).item()

    prev_runs = RunHistory()
    for i in range(initial_data_x.size(0)):
        prev_runs.add(config=initial_data_x_smac_format[i], cost=initial_data_y[i].item(), time=0,
                      status=StatusType.SUCCESS, seed=exp_id)

    name_tag = generate_exp_tag(blackbox_evaluator=blackbox_evaluator, exp_id=exp_id)
    save_file = os.path.join(exp_dir_root(), name_tag)

    print('Began    at ' + datetime.now().strftime("%H:%M:%S"))
    scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval,
                         "cs": cs, "deterministic": "false", 'output_dir': save_file})
    smac = SMAC4HPO(scenario=scenario, tae_runner=evaluate, runhistory=prev_runs, initial_design=RandomConfigurations)
    smac.solver.intensifier.use_pynisher = False
    smac.optimize()

    evaluations, optimum = evaluations_from_smac(smac)
    print('Finished at ' + datetime.now().strftime("%H:%M:%S"))

    if save_dir is not None:
        with open(os.path.join(save_dir, 'SMAC_%s_[R%d].pkl' % (exp_type, exp_id)), 'wb') as f:
            pickle.dump({'evaluations': evaluations, 'optimum': optimum}, f)

    return optimum


def multiple_runs(exp_type, n_eval):
    n_runs = 5
    runs = None
    for exp_id in range(n_runs):
        optimum = smac_wrapper(n_eval, exp_type=exp_type, exp_id=exp_id)
        if runs is None:
            runs = optimum.reshape(-1, 1)
        else:
            runs = np.hstack([runs, optimum.reshape(-1, 1)])
        print('exp_id : %d has been finished' % exp_id)

    mean = np.mean(runs, axis=1)
    std = np.std(runs, axis=1)
    with open(os.path.join(exp_dir_root(), exp_type + '_baseline_result_smac.pkl'), 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)

    return mean, std


def collect_individual_runs():
    filename_list = ['SMAC_XGBFashionMNIST_[R%d].pkl' % i for i in range(5)]
    runs = None
    for filename in filename_list:
        with open(os.path.join(exp_dir_root(), filename), 'rb') as f:
            data = pickle.load(f)
        if runs is None:
            runs = data['optimum'].reshape(-1, 1)
        else:
            runs = np.hstack([runs, data['optimum'].reshape(-1, 1)])
    mean = np.mean(runs, axis=1)
    std = np.std(runs, axis=1)
    with open(os.path.join(exp_dir_root(), 'XGBFashionMNIST_baseline_result_smac.pkl'), 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)


def evaluations_from_smac(smac):
    evaluations = np.array([elm.cost for elm in smac.get_runhistory().data.values()])
    n_evals = evaluations.size
    optimum = np.zeros((n_evals, ))
    for i in range(n_evals):
        optimum[i] = np.min(evaluations[:i+1])
    return evaluations, optimum


if __name__ == '__main__':
    collect_individual_runs()
    # parser = argparse.ArgumentParser(description='SMAC')
    # parser.add_argument('--exp_type', dest='exp_type', type=str, default=None)
    # parser.add_argument('--exp_id', dest='exp_id', type=int, default=None, help='[0,1,2,3,4]')
    #
    # args = parser.parse_args()
    # smac_wrapper(n_eval=100, exp_type=args.exp_type, exp_id=args.exp_id, save_dir=exp_dir_root())
