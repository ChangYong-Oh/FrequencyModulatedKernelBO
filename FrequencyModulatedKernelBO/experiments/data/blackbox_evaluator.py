from typing import Optional, List, Tuple

import numpy as np

import torch

from FrequencyModulatedKernelBO.experiments.data.utils import MAX_RANDOM_SEED,\
    path_graph_fourier, complete_graph_fourier


class BlackboxEvaluator(object):
    def __init__(self, list_of_continuous: List, list_of_ordinal: List[int], list_of_nominal: List[int],
                 n_init_data: int, init_data_seed: Optional[int] = None, eval_init: bool = False):
        """
        :param list_of_continuous : a positive integer, the number of continuous variables
        :param list_of_ordinal : a list of positive integer,
               list_of_ordinal[i] = C means i-th discrete ordinal variables has C categories
        :param list_of_nominal : a list of positive integer,
               list_of_nominal[i] = C means i-th discrete nominal variables has C categories
        :param n_init_data : the number of variables interacting with a variable
        :param init_data_seed : a positive integer, using None determines random_seed from 3 other arguments
               so that it becomes deterministic
        """
        self.list_of_contiuous = list_of_continuous
        self.list_of_ordinal = list_of_ordinal
        self.list_of_nominal = list_of_nominal
        self.n_continuous = len(list_of_continuous)
        self.n_ordinal = len(list_of_ordinal)
        self.n_nominal = len(list_of_nominal)
        self.n_variable = self.n_continuous + self.n_ordinal + self.n_nominal
        self.init_data_seed = init_data_seed
        self.minimize = None

        self.optimized_features = dict()
        self.fourier_frequency = dict()
        self.fourier_basis = dict()
        for d in range(self.n_continuous):
            assert len(self.list_of_contiuous[d]) == 2
            self.optimized_features[d] = self.list_of_contiuous[d]
        self.shift_ordinal = self.n_continuous
        for d, ordinal in enumerate(list_of_ordinal):
            adj_mat, fourier_freq, fourier_basis = path_graph_fourier(ordinal)
            self.optimized_features[d + self.shift_ordinal] = adj_mat
            self.fourier_frequency[d + self.shift_ordinal] = fourier_freq
            self.fourier_basis[d + self.shift_ordinal] = fourier_basis
        self.shift_nominal = self.n_continuous + self.n_ordinal
        for d, nominal in enumerate(list_of_nominal):
            adj_mat, fourier_freq, fourier_basis = complete_graph_fourier(nominal)
            self.optimized_features[d + self.shift_nominal] = adj_mat
            self.fourier_frequency[d + self.shift_nominal] = fourier_freq
            self.fourier_basis[d + self.shift_nominal] = fourier_basis

        self.initial_data = self.generate_initial_data(n_init_data, eval_init)
        self.info_str = None

    def state_dict(self):
        return {'name': self.__class__.__name__}

    def generate_initial_data(self, n: int, eval_init: bool):
        initial_data_x = np.zeros((n, self.n_variable))
        seed_list = np.random.RandomState(self.init_data_seed).randint(0, MAX_RANDOM_SEED - 1, self.n_variable)
        for d in range(self.n_continuous):
            low, high = self.optimized_features[d]
            initial_data_x[:, d] = np.random.RandomState(seed_list[d]).uniform(low, high, n)
        for d, ordinal in enumerate(self.list_of_ordinal):
            initial_data_x[:, d + self.shift_ordinal] = \
                np.random.RandomState(seed_list[d + self.shift_ordinal]).randint(0, ordinal, n)
        for d, nominal in enumerate(self.list_of_nominal):
            initial_data_x[:, d + self.shift_nominal] = \
                np.random.RandomState(seed_list[d + self.shift_nominal]).randint(0, nominal, n)
        initial_data_x = torch.from_numpy(initial_data_x.astype(np.float32))
        if eval_init:
            initial_data_y = self.evaluate(initial_data_x)
            return initial_data_x, initial_data_y
        else:
            return initial_data_x, None

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
