from typing import Optional

import os
import pickle
import argparse
import numpy as np

import torch

from hyperopt import hp, base
from hyperopt import fmin, tpe

from FrequencyModulatedKernelBO.experiments.config import exp_dir_root
from FrequencyModulatedKernelBO.experiments.data import Func2CEvaluator, Func3CEvaluator, Ackley5CEvaluator, \
    SVMBostonEvaluator, XGBFashionMNISTEvaluator


def tpe_wrapper(n_eval, exp_type, exp_id, save_dir: Optional[str] = None):
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

    space = []
    for i, (lower, upper) in enumerate(blackbox_evaluator.list_of_contiuous):
        continuous_var = hp.uniform('x' + str(i + 1).zfill(2), low=lower, high=upper)
        space.append(continuous_var)
    for i, n_values in enumerate(blackbox_evaluator.list_of_ordinal):
        ordinal_var = hp.randint('x' + str(i + blackbox_evaluator.shift_ordinal + 1).zfill(2), low=0, high=n_values)
        space.append(ordinal_var)
    for i, n_values in enumerate(blackbox_evaluator.list_of_nominal):
        nominal_var = hp.choice('x' + str(i + blackbox_evaluator.shift_nominal + 1).zfill(2),
                                options=[int(elm) for elm in range(n_values)])
        space.append(nominal_var)

    if exp_type == 'XGBFashionMNIST':
        with open(os.path.join(exp_dir_root(), '%s_[R%d].pkl' % (exp_type, exp_id)), 'rb') as f:
            data = torch.load(f)['data']
        initial_data_x = data['eval_x']
        initial_data_y = data['eval_y']
    else:
        initial_data_x, initial_data_y = blackbox_evaluator.initial_data
    initial_data_x_tpe_format = []
    for i in range(initial_data_x.size(0)):
        data_x_tpe_format = dict()
        for j in range(n_continuous):
            data_x_tpe_format['x' + str(j + 1).zfill(2)] = initial_data_x[i, j].item()
        for j in range(n_continuous, n_continuous + n_ordinal):
            data_x_tpe_format['x' + str(j + 1).zfill(2)] = int(initial_data_x[i, j].item())
        for j in range(n_continuous + n_ordinal, n_continuous + n_ordinal + n_nominal):
            data_x_tpe_format['x' + str(j + 1).zfill(2)] = int(initial_data_x[i, j].item())
        initial_data_x_tpe_format.append(data_x_tpe_format)

    def evaluate(x):
        return blackbox_evaluator(torch.tensor(x).view(1, -1)).item()

    trials = base.Trials()
    fmin(evaluate, space, algo=tpe.suggest, max_evals=n_eval, points_to_evaluate=initial_data_x_tpe_format,
         trials=trials)
    evaluations, optimum = evaluations_from_trials(trials, initial_data_y.numpy())

    if save_dir is not None:
        with open(os.path.join(save_dir, 'TPE_%s_[R%d].pkl' % (exp_type, exp_id)), 'wb') as f:
            pickle.dump({'evaluations': evaluations, 'optimum': optimum}, f)

    return optimum


def multiple_runs(exp_type, n_eval):
    print('Optimizing %s' % exp_type)
    runs = None
    for exp_id in range(5):
        optimum = tpe_wrapper(n_eval, exp_type, exp_id)
        if runs is None:
            runs = optimum.reshape(-1, 1)
        else:
            runs = np.hstack([runs, optimum.reshape(-1, 1)])

    print('\nOptimized %s' % exp_type)

    mean = np.mean(runs, axis=1)
    std = np.std(runs, axis=1)
    tpe_file = open(os.path.join(exp_dir_root(), exp_type + '_baseline_result_tpe.pkl'), 'wb')
    pickle.dump({'mean': mean, 'std': std}, tpe_file)
    tpe_file.close()

    return np.mean(runs, axis=1), np.std(runs, axis=1)


def collect_individual_runs():
    filename_list = ['TPE_XGBFashionMNIST_[R%d].pkl' % i for i in range(5)]
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
    tpe_file = open(os.path.join(exp_dir_root(), 'XGBFashionMNIST_baseline_result_tpe.pkl'), 'wb')
    pickle.dump({'mean': mean, 'std': std}, tpe_file)
    tpe_file.close()


def evaluations_from_trials(trials, init_y):
    n_trials = len(trials.trials) + init_y.flatten().shape[0]
    evaluations = np.concatenate([init_y.flatten(),
                                  np.array([trials.trials[i]['result']['loss'] for i in range(len(trials.trials))])])
    optimum = np.zeros((n_trials, ))
    for i in range(n_trials):
        optimum[i] = np.min(evaluations[:i+1])
    return evaluations, optimum


if __name__ == '__main__':
    collect_individual_runs()
    # parser = argparse.ArgumentParser(description='TPE')
    # parser.add_argument('--exp_type', dest='exp_type', type=str, default=None)
    # parser.add_argument('--exp_id', dest='exp_id', type=int, default=None, help='[0,1,2,3,4]')
    #
    # args = parser.parse_args()
    # tpe_wrapper(n_eval=100, exp_type=args.exp_type, exp_id=args.exp_id, save_dir=exp_dir_root())