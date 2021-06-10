from typing import Optional, Tuple, List

import time
import os
import json
import pickle
import copy
import math
import datetime
import itertools
import argparse

import numpy as np
from scipy.special import binom

import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
from hpbandster.core.result import Result, json_result_logger
from hpbandster.examples.commons import MyWorker

from FrequencyModulatedKernelBO.experiments.nas.nasnet import NASNet
from FrequencyModulatedKernelBO.experiments.nas.utils import compute_flops
from FrequencyModulatedKernelBO.experiments.data.exp_nas_nasnet import evaluate_nasnet_hyperparams, \
    LOG_LR_RNG, MOMENTUM_RNG, LOG_WEIGHT_DECAY_RNG, GAMMA_RNG, MILESTONE_RATIO1_RNG, MILESTONE_RATIO2_RNG, \
    NASNET_OPS, N_STATES, N_CHANNEL_LIST, MAX_FLOP_BLOCK_OUTPUT_STATE, MAX_FLOP_BLOCK_ARCHITECTURE, HP_FashionMNIST
from FrequencyModulatedKernelBO.experiments.bo_nas_nasnet import NASNET_EXP_DIR


class NASNetWorker(Worker):

    def __init__(self, data: str, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

        self.list_of_categories = []
        self.category_encoding: List = []
        for i in range(N_STATES * 2):
            n_cats = len(NASNET_OPS)
            self.list_of_categories.append(n_cats)
            self.category_encoding.append(NASNET_OPS[:])
        for i in range(2, N_STATES + 1):
            n_cats = int(binom(i + 1, 2))
            self.list_of_categories.append(n_cats)
            input_pairs = list(itertools.combinations(range(-1, i), 2))
            for pair in input_pairs:
                assert pair[0] < pair[1]
            self.category_encoding.append(input_pairs)
        self.list_of_categories[-1] -= 1
        self.category_encoding[-1].remove((-1, 0))
        self.list_of_categories.append(N_STATES - 1)
        self.category_encoding.append(list(range(1, N_STATES)))

        self.data = data
        self._hyper_params = globals()['HP_' + self.data]
        self.max_flops = compute_flops(
            m=NASNet(block_architecture=MAX_FLOP_BLOCK_ARCHITECTURE, block_output_state=MAX_FLOP_BLOCK_OUTPUT_STATE,
                     input_dim=self._hyper_params['input_dim'], n_channel_list=N_CHANNEL_LIST,
                     num_classes=self._hyper_params['num_classes']),
            input_dim=self._hyper_params['input_dim'])

    def compute(self, config, budget, **kwargs):
        assert 0.5 < budget < self._hyper_params['max_epoch'] + 0.5
        n_epochs = round(budget)
        assert 1 <= n_epochs <= 25
        lr = np.exp(config['log_lr'])
        momentum = config['momentum']
        weight_decay = np.exp(config['log_weight_decay'])
        gamma = config['gamma']
        milestone_ratio1 = config['milestone_ratio1']
        milestone_ratio2 = config['milestone_ratio2']
        milestones_ratio = [milestone_ratio1, milestone_ratio1 + (1 - milestone_ratio1) * milestone_ratio2]

        block_architecture = []
        for i in range(N_STATES):
            op1 = NASNET_OPS[int(config['op%02d' % (2 * i)])]
            op2 = NASNET_OPS[int(config['op%02d' % (2 * i + 1)])]
            if i > 0:
                input1, input2 = self.category_encoding[2 * N_STATES + i - 1][int(config['cell%02dinput' % (i + 1)])]
            else:
                input1, input2 = -1, 0
            block_architecture.append(((input1, op1), (input2, op2)))
            print('STATE(%d) : INPUT1(%2d -> %18s) + INPUT2(%2d -> %18s)'
                  % (i + 1, input1, op1, input2, op2))
        block_output_state = int(self.category_encoding[-1][int(config['blockoutput'])])

        valid_error, flops = evaluate_nasnet_hyperparams(
            block_architecture=block_architecture, block_output_state=block_output_state,
            data_type=self.data,
            input_dim=self._hyper_params['input_dim'], n_channel_list=N_CHANNEL_LIST,
            num_classes=self._hyper_params['num_classes'],
            max_epoch=n_epochs, batch_size=self._hyper_params['batch_size'],
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            gamma=gamma, milestones_ratio=milestones_ratio,
            n_repeated_eval=self._hyper_params['n_repeated_eval'], milestones_epoch=self._hyper_params['max_epoch'])
        print('[%s] Evaluation(s) finished' % datetime.datetime.now().strftime('%H:%M:%S'))

        print('Classification Error : %.4f / FLOPs ratio : %.4f' % (valid_error, flops / self.max_flops))
        print('\n' * 3)
        log_loss = math.log(valid_error + 0.02 * flops / self.max_flops)
        return {'loss': float(log_loss), 'info': log_loss}

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('log_lr',
                                          lower=LOG_LR_RNG[0], upper=LOG_LR_RNG[1]))
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('momentum',
                                          lower=MOMENTUM_RNG[0], upper=MOMENTUM_RNG[1]))
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('log_weight_decay',
                                          lower=LOG_WEIGHT_DECAY_RNG[0], upper=LOG_WEIGHT_DECAY_RNG[1]))
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('gamma',
                                          lower=GAMMA_RNG[0], upper=GAMMA_RNG[1]))
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('milestone_ratio1',
                                          lower=MILESTONE_RATIO1_RNG[0], upper=MILESTONE_RATIO1_RNG[1]))
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('milestone_ratio2',
                                          lower=MILESTONE_RATIO2_RNG[0], upper=MILESTONE_RATIO2_RNG[1]))
        list_of_categories = []
        category_encoding = []
        for i in range(N_STATES * 2):
            n_cats = len(NASNET_OPS)
            list_of_categories.append(n_cats)
            category_encoding.append(NASNET_OPS[:])
            config_space.add_hyperparameter(
                CS.CategoricalHyperparameter('op%02d' % i, [str(int(elm)) for elm in range(n_cats)]))
        for i in range(2, N_STATES + 1):
            n_cats = int(binom(i + 1, 2))
            list_of_categories.append(n_cats)
            input_pairs = list(itertools.combinations(range(-1, i), 2))
            for pair in input_pairs:
                assert pair[0] < pair[1]
            category_encoding.append(input_pairs)
            if i < N_STATES:
                config_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('cell%02dinput' % i, [str(int(elm)) for elm in range(n_cats)]))
        list_of_categories[-1] -= 1
        category_encoding[-1].remove((-1, 0))
        n_cats = list_of_categories[-1]
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter('cell%02dinput' % N_STATES, [str(int(elm)) for elm in range(n_cats)]))
        list_of_categories.append(N_STATES - 1)
        category_encoding.append(list(range(1, N_STATES)))
        n_cats = list_of_categories[-1]
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter('blockoutput', [str(int(elm)) for elm in range(n_cats)]))
        return config_space


def bohb_exp_name(data: str, seed: int, worker_name: str, idx: int):
    return 'BOHB_%s_%s_[R%d]_%04d' % (worker_name, data, seed, idx)


def bohb_next_exp_dir(dirname):
    _, worker_name, data, seed, idx = dirname.split('_')
    return bohb_exp_name(data=data, seed=int(seed[-2:-1]), worker_name=worker_name, idx=int(idx) + 1)


def last_file_in_exp(dirname: str):
    max_iteration = -1
    max_iteration_dir = None
    for elm in os.listdir(os.path.join(NASNET_EXP_DIR, dirname)):
        if os.path.isfile(os.path.join(NASNET_EXP_DIR, dirname, elm)) and os.path.splitext(elm)[1] == '.pkl':
            iteration_str = os.path.splitext(elm)[0].split('_')[-1]
            assert len(iteration_str) == 4
            iteration = int(iteration_str)
            if iteration > max_iteration:
                max_iteration = iteration
                max_iteration_dir = elm

    return max_iteration_dir, max_iteration


def bohb_continue(prev_exp_dir: str, n_iterations: int = 1, host: str = '127.0.0.1'):
    last_file, last_iteration = last_file_in_exp(prev_exp_dir)
    worker_name, data, seed, idx = prev_exp_dir.split('_')[1:]
    run_id = '_'.join([data, seed, '%04d' % (last_iteration + 1)])
    curr_exp_dir = bohb_next_exp_dir(prev_exp_dir)

    NS = hpns.NameServer(run_id=run_id, host=host, port=None)
    NS.start()

    nameserver = host
    if worker_name.upper() == 'NASNET':
        w = NASNetWorker(data=data, sleep_interval=0, nameserver=nameserver, run_id=run_id)
        min_budget = 0.5 + 1e-6
        max_budget = w._hyper_params['max_epoch'] + 0.5 - 1e-6
    elif worker_name.upper() == 'MY':
        w = MyWorker(sleep_interval=0, host=host, nameserver=nameserver, run_id=run_id)
        min_budget = 0.5 + 1e-6
        max_budget = 25 + 0.5 - 1e-6
    else:
        raise NotImplementedError
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=os.path.join(NASNET_EXP_DIR, curr_exp_dir), overwrite=False)
    previous_run = hpres.logged_results_to_HBS_result(os.path.join(NASNET_EXP_DIR, prev_exp_dir))
    bohb = BOHB(
        configspace=w.get_configspace(), run_id=run_id, nameserver=nameserver, result_logger=result_logger,
        min_budget=min_budget, max_budget=max_budget, previous_result=previous_run)
    bohb.run(n_iterations=n_iterations)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


def bohb_init(data: str, seed: int, worker_name: str, n_iterations: int = 1, host: str = '127.0.0.1'):
    exp_dir = bohb_exp_name(data=data, seed=seed, worker_name=worker_name, idx=0)
    run_id = exp_dir

    NS = hpns.NameServer(run_id=run_id, host=host, port=None)
    NS.start()

    log_directory = os.path.join(NASNET_EXP_DIR, exp_dir)

    nameserver = host
    if worker_name.upper() == 'NASNET':
        w = NASNetWorker(data=data, sleep_interval=0, nameserver=nameserver, run_id=run_id)
        min_budget = 0.5 + 1e-6
        max_budget = w._hyper_params['max_epoch'] + 0.5 - 1e-6
    elif worker_name.upper() == 'MY':
        w = MyWorker(sleep_interval=0, host=host, nameserver=nameserver, run_id=run_id)
        min_budget = 0.5 + 1e-6
        max_budget = 25 + 0.5 - 1e-6
    else:
        raise NotImplementedError
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=log_directory, overwrite=False)

    bohb = BOHB(
        configspace=w.get_configspace(), run_id=run_id, nameserver=nameserver, result_logger=result_logger,
        min_budget=min_budget, max_budget=max_budget)
    bohb.run(n_iterations=n_iterations)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    return exp_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BOHB')
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    args = parser.parse_args()

    exp_dir_ = bohb_init(data='FashionMNIST', seed=args.seed, worker_name='NASNET', n_iterations=1000,
                         host='127.0.0.%d' % (args.seed + 1))
