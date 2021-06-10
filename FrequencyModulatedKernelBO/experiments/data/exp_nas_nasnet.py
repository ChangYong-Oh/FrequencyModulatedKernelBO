from typing import Tuple, List, Optional

import datetime
import itertools
import sys
import warnings
import time

import numpy as np
from scipy.special import binom

import torch
import torch.optim
import torch.multiprocessing

from FrequencyModulatedKernelBO.experiments.data.blackbox_evaluator import BlackboxEvaluator
from FrequencyModulatedKernelBO.experiments.data.utils import complete_graph_fourier, path_graph_fourier, graph_fourier
from FrequencyModulatedKernelBO.experiments.config import MAX_RANDOM_SEED
from FrequencyModulatedKernelBO.experiments.nas.data_loader import load_data
from FrequencyModulatedKernelBO.experiments.nas.nasnet import NASNet, NASNET_OPS
from FrequencyModulatedKernelBO.experiments.nas.utils import compute_flops


N_INIT_NAS_NASNET = 10
N_STATES = 4
N_CHANNEL_LIST = [8, 16, 32]

N_REPEATED_EVAL = 4

LOG_LR_RNG = (np.log(0.001), np.log(0.1))
MOMENTUM_RNG = (0.8, 1.0)
LOG_WEIGHT_DECAY_RNG = (np.log(1e-6), np.log(1e-2))
GAMMA_RNG = (0.1, 0.9)
MILESTONE_RATIO1_RNG = (0, 1)
MILESTONE_RATIO2_RNG = (0, 1)


HP_CIFAR = {
    'input_dim': [3, 32, 32],
    'batch_size': 64,
    'max_epoch': 50,
    'n_repeated_eval': N_REPEATED_EVAL,
    }
HP_CIFAR10 = {'num_classes': 10, **HP_CIFAR}
HP_CIFAR100 = {'num_classes': 100, **HP_CIFAR}

HP_FashionMNIST = {
    'num_classes': 10,
    'input_dim': [1, 28, 28],
    'batch_size': 32,
    'max_epoch': 25,
    'n_repeated_eval': N_REPEATED_EVAL,
}


EXP_DATA = 'FashionMNIST'
HYPER_PARAMS = globals()['HP_' + EXP_DATA]
MAX_FLOP_BLOCK_ARCHITECTURE = []
HEAVIEST_OP = 'Conv5by5'
assert HEAVIEST_OP in NASNET_OPS
for s in range(N_STATES):
    MAX_FLOP_BLOCK_ARCHITECTURE.append(((s - 1, HEAVIEST_OP), (s, HEAVIEST_OP)))
MAX_FLOP_BLOCK_OUTPUT_STATE = N_STATES - 1


def _multiprocessing_wrapper(i: int, model: NASNet, optimizer, scheduler, train_loader, eval_loader, max_epoch: int,
                             result_queue):
    warnings.filterwarnings('ignore', category=FutureWarning)
    print('Multprocessing %d' % i)
    sys.stdout.flush()
    while True:
        try:
            eval_error = _train_eval_nasnet(
                model=model, optimizer=optimizer, scheduler=scheduler,
                train_loader=train_loader, eval_loader=eval_loader, max_epoch=max_epoch, verbose=False)
            break
        except RuntimeError as err:
            print('RuntimeError %s occurred' % err)
            continue
    result_queue.put(eval_error)


def _train_eval_nasnet(model: NASNet, optimizer, scheduler,
                       train_loader, eval_loader, max_epoch: int, verbose: bool = True):
    warnings.filterwarnings('ignore', category=UserWarning)
    model.train()
    model_device = next(model.parameters()).device
    use_gpu = next(model.parameters()).is_cuda

    if verbose:
        print('[%s]%3d epoch : Training begins' % (datetime.datetime.now().strftime('%H:%M:%S'), max_epoch))
    model.initialize_params()
    for e in range(max_epoch):
        running_loss = 0
        for input_batch, output_batch in train_loader:
            if use_gpu:
                input_batch = input_batch.cuda(device=model_device)
                output_batch = output_batch.cuda(device=model_device)
            pred_batch = model(input_batch)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(pred_batch, output_batch, reduction='mean')
            loss.backward()
            optimizer.step()
            running_loss = 0.95 * running_loss + 0.05 * loss.item()
        if verbose and (e + 1) % 1 == 0:
            print('[%s]%3d epoch : Loss : %+.6E, lr : %.8f' % (datetime.datetime.now().strftime('%H:%M:%S'),
                                                               e + 1, running_loss, scheduler.get_last_lr()[0]))
        scheduler.step()

    model.eval()
    n_data = 0
    n_correct = 0
    for input_batch, output_batch in eval_loader:
        if use_gpu:
            input_batch = input_batch.cuda(device=model_device)
            output_batch = output_batch.cuda(device=model_device)
        pred_batch = model(input_batch)
        n_data += input_batch.size(0)
        n_correct += torch.sum(torch.argmax(pred_batch, dim=1) == output_batch).item()

    return 1 - n_correct / n_data


def evaluate_nasnet_hyperparams(block_architecture: List[Tuple[Tuple[int, str], Tuple[int, str]]],
                                block_output_state: int,
                                data_type: str,
                                input_dim: List[int],
                                n_channel_list: List[int],
                                num_classes: int,
                                max_epoch: int,
                                batch_size: int,
                                lr: float,
                                momentum: float,
                                weight_decay: float,
                                gamma: float,
                                milestones_ratio: List[float],
                                n_repeated_eval: int,
                                milestones_epoch: Optional[int] = None):
    use_gpu = True and torch.cuda.is_available()
    pin_memory = use_gpu

    train_loader, valid_loader, test_loader = load_data(data_type=data_type, batch_size=batch_size,
                                                        pin_memory=pin_memory)
    model = NASNet(block_architecture=block_architecture, block_output_state=block_output_state,
                   input_dim=input_dim, n_channel_list=n_channel_list, num_classes=num_classes)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay,
                                momentum=momentum)
    if milestones_epoch is None:
        milestones_epoch = max_epoch
    milestones = [int(milestones_epoch * m) for m in milestones_ratio]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    print('On %s data set (%d training, %d validation), %d epochs with batch size of %d'
          % (data_type, len(train_loader.sampler), len(valid_loader.sampler), max_epoch, batch_size))
    print('Each cell has following filter size : %s' % '-'.join([str(elm) for elm in n_channel_list]))
    print('Learning rate scheduling \n\t1) %3d~%3d : %.8f \n\t2) %3d~%3d : %.8f \n\t3) %3d~%3d : %.8f'
          % (0, milestones[0], lr, milestones[0], milestones[1], lr * gamma, milestones[1], milestones_epoch, lr * gamma ** 2))

    if use_gpu:
        model.cuda()

    n_processes = n_repeated_eval
    try:
        torch.multiprocessing.set_start_method(method='spawn')
    except RuntimeError:
        pass
    result_queue = torch.multiprocessing.Queue()
    processes = []
    for p in range(n_processes):
        p = torch.multiprocessing.Process(target=_multiprocessing_wrapper,
                                          args=(p, model, optimizer, scheduler,
                                                train_loader, valid_loader, max_epoch, result_queue))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    result = [result_queue.get() for _ in range(n_processes)]
    print('All %d evaluations %s' % (n_processes, ', '.join(['%.6f' % r for r in result])))
    valid_error = np.mean(result)
    flops = compute_flops(m=model.cpu(), input_dim=input_dim)
    return valid_error, flops


def _nasnet_connectivity_graph_fourier(encoding: List[Tuple[int, int]]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_vertices = len(encoding)
    for e in encoding:
        assert e[0] < e[1]
    adjacency_matrix = torch.zeros(n_vertices, n_vertices)
    for i in range(1, n_vertices):
        for j in range(i):
            if (encoding[i][0] == encoding[j][0] and abs(encoding[i][1] - encoding[j][1]) == 1) \
                    or (abs(encoding[i][0] - encoding[j][0]) == 1 and encoding[i][1] == encoding[j][1]) \
                    or (encoding[i][0] == encoding[j][1] and abs(encoding[i][1] - encoding[j][0]) == 1) \
                    or (abs(encoding[i][0] - encoding[j][1]) == 1 and encoding[i][1] == encoding[j][0]):
            # if len(set(encoding[i]).intersection(set(encoding[j]))) > 0:
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    fourier_frequency, fourier_basis = graph_fourier(adjacency_matrix)
    return adjacency_matrix, fourier_frequency, fourier_basis


class NASNetEvaluator(object):
    def __init__(self, data: str, init_data_seed: Optional[int] = None, eval_init: bool = False):
        """
        :param init_data_seed : a positive integer, using None determines random_seed from 3 other arguments
               so that it becomes deterministic
        """
        self.data = data
        self._hyper_params = globals()['HP_' + self.data]

        self.max_flops = compute_flops(
            m=NASNet(block_architecture=MAX_FLOP_BLOCK_ARCHITECTURE, block_output_state=MAX_FLOP_BLOCK_OUTPUT_STATE,
                     input_dim=self._hyper_params['input_dim'], n_channel_list=N_CHANNEL_LIST,
                     num_classes=self._hyper_params['num_classes']),
            input_dim=self._hyper_params['input_dim'])

        self.n_continuous_vars = 6
        self.optimized_features = dict()
        self.optimized_features[0] = LOG_LR_RNG
        self.optimized_features[1] = MOMENTUM_RNG  # momentum
        self.optimized_features[2] = LOG_WEIGHT_DECAY_RNG  # log weight_decay
        self.optimized_features[3] = GAMMA_RNG  # gamma (lr scheduling)
        self.optimized_features[4] = MILESTONE_RATIO1_RNG
        self.optimized_features[5] = MILESTONE_RATIO2_RNG

        self.list_of_categories = []
        self.category_encoding = []
        for i in range(N_STATES * 2):
            self.list_of_categories.append(len(NASNET_OPS))
            self.category_encoding.append(NASNET_OPS[:])
        for i in range(2, N_STATES + 1):
            self.list_of_categories.append(int(binom(i + 1, 2)))
            input_pairs = list(itertools.combinations(range(-1, i), 2))
            for pair in input_pairs:
                assert pair[0] < pair[1]
            self.category_encoding.append(input_pairs)
        self.list_of_categories[-1] -= 1
        self.category_encoding[-1].remove((-1, 0))
        self.list_of_categories.append(N_STATES - 1)
        self.category_encoding.append(list(range(1, N_STATES)))

        self.fourier_frequency = dict()
        self.fourier_basis = dict()
        for d, (category, encoding) in enumerate(zip(self.list_of_categories, self.category_encoding)):
            if d < 2 * N_STATES:
                adj_mat, fourier_freq, fourier_basis = complete_graph_fourier(category)
            elif 2 * N_STATES <= d < 3 * N_STATES - 1:
                adj_mat, fourier_freq, fourier_basis = _nasnet_connectivity_graph_fourier(encoding)
            elif d == 3 * N_STATES - 1:
                # adj_mat, fourier_freq, fourier_basis = complete_graph_fourier(category)
                adj_mat, fourier_freq, fourier_basis = path_graph_fourier(category)
            else:
                raise ValueError
            self.optimized_features[d + self.n_continuous_vars] = adj_mat
            self.fourier_frequency[d + self.n_continuous_vars] = fourier_freq
            self.fourier_basis[d + self.n_continuous_vars] = fourier_basis

        self.init_data_seed = init_data_seed
        self.n_init_data = N_INIT_NAS_NASNET
        if eval_init:
            self.initial_data = self.generate_initial_data(self.n_init_data)
        self.info_str = 'NASNASNET'

    def state_dict(self):
        return {'name': self.__class__.__name__, 'data': self.data, 'init_data_seed': self.init_data_seed}

    def generate_initial_data(self, n) -> Tuple[torch.Tensor, torch.Tensor]:
        ndim = self.n_continuous_vars + len(self.list_of_categories)
        initial_data_x = np.zeros((n, ndim))
        seed_list = np.random.RandomState(self.init_data_seed).randint(0, MAX_RANDOM_SEED - 1, ndim)
        for d in range(self.n_continuous_vars):
            low, high = self.optimized_features[d]
            initial_data_x[:, d] = np.random.RandomState(seed_list[d]).uniform(low, high, n)
        for d, category in enumerate(self.list_of_categories):
            initial_data_x[:, d + self.n_continuous_vars] = \
                np.random.RandomState(seed_list[d + self.n_continuous_vars]).randint(0, category, n)
        initial_data_x = torch.from_numpy(initial_data_x.astype(np.float32))
        initial_data_y = initial_data_x.new_empty(size=(initial_data_x.size(0), 1))
        for i in range(initial_data_x.size(0)):
            start_time = time.time()
            initial_data_y[i] = self.evaluate(initial_data_x[i])
            print('%6d seconds to evaluate' % int(time.time() - start_time))
        return initial_data_x, initial_data_y

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.numel() == self.n_continuous_vars + len(self.list_of_categories)
        if x.dim() == 2:
            assert x.size(0) == 1
            x = x.view(-1)
        lr = np.exp(x[0].item())
        momentum = x[1].item()
        weight_decay = np.exp(x[2].item())
        gamma = x[3].item()
        milestone_ratio1 = x[4].item()
        milestone_ratio2 = x[5].item()
        milestones_ratio = [milestone_ratio1, milestone_ratio1 + (1 - milestone_ratio1) * milestone_ratio2]

        print('=' * 80)
        print('DATA %s[R%d]' % (self.data, self.init_data_seed))
        print('Continuous variables %s' % ', '.join(['%+.4E' % x[c] for c in range(self.n_continuous_vars)]))
        print('Discrete variables %s' % ', '.join(['%2d' % x[self.n_continuous_vars + d]
                                                   for d in range(len(self.list_of_categories))]))
        print('BLOCK-INPUTS :  INPUT1 (%2d), INPUT2 (%2d)' % (-1, 0))
        block_architecture = []
        for i in range(N_STATES):
            op1 = NASNET_OPS[x[self.n_continuous_vars + i * 2].long().item()]
            op2 = NASNET_OPS[x[self.n_continuous_vars + i * 2 + 1].long().item()]
            if i > 0:
                input1, input2 = self.category_encoding[2 * N_STATES + i - 1][
                    x[self.n_continuous_vars + 2 * N_STATES + i - 1].long().item()]
            else:
                input1, input2 = -1, 0
            block_architecture.append(((input1, op1), (input2, op2)))
            print('STATE(%d) : INPUT1(%2d -> %18s) + INPUT2(%2d -> %18s)'
                  % (i + 1, input1, op1, input2, op2))
        block_output_state = int(self.category_encoding[-1][x[-1].long().item()])
        print('BLOCK-OUTPUTS : OUTPUT1(%2d), OUTPUT2(%2d)' % (block_output_state, N_STATES))

        print('DATA %s[R%d]' % (self.data, self.init_data_seed))  
        print('[%s] %d Evaluation(s) began in parallel'
              % (datetime.datetime.now().strftime('%H:%M:%S'), self._hyper_params['n_repeated_eval']))
        valid_error, flops = evaluate_nasnet_hyperparams(
            block_architecture=block_architecture, block_output_state=block_output_state,
            data_type=self.data,
            input_dim=self._hyper_params['input_dim'], n_channel_list=N_CHANNEL_LIST,
            num_classes=self._hyper_params['num_classes'],
            max_epoch=self._hyper_params['max_epoch'], batch_size=self._hyper_params['batch_size'],
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            gamma=gamma, milestones_ratio=milestones_ratio,
            n_repeated_eval=self._hyper_params['n_repeated_eval'])
        print('[%s] Evaluation(s) finished' % datetime.datetime.now().strftime('%H:%M:%S'))

        print('Classification Error : %.4f / FLOPs ratio : %.4f' % (valid_error, flops / self.max_flops))
        print('\n' * 3)
        return (valid_error + 0.02 * flops / self.max_flops) * torch.ones(1, 1)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


if __name__ == '__main__':
    evalulator = NASNetEvaluator(data='FashionMNIST', init_data_seed=0, eval_init=True)




