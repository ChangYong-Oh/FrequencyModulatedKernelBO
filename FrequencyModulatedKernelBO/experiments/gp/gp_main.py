import os
import math
import argparse
from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

import torch

from FrequencyModulatedKernelBO.experiments.data.utils import complete_graph_fourier
from FrequencyModulatedKernelBO.experiments.gp.gp_data_load import load_uci_data
from FrequencyModulatedKernelBO.experiments.gp_models import MixedVariableGPOptimize

DATA_SEED_LIST = list(range(20))
KERNEL_TYPE_LIST = ['AdditiveDiffusion', 'ProductDiffusion', 'ModulatedDiffusion',
                    'AdditiveLaplacian', 'ProductLaplacian', 'ModulatedLaplacian']


def read_data():
    for filename in ['REG1.pth', 'REG2.pth', 'REG3.pth']:
        result_dict_dict = torch.load(filename)
        print(filename)
        kernel_type_str_list = []
        nll_str_list = []
        rmse_str_list = []
        for kernel_type in KERNEL_TYPE_LIST:
            test_nll_list = [result_dict_dict[elm][kernel_type][0].item() for elm in DATA_SEED_LIST]
            test_rmse_list = [result_dict_dict[elm][kernel_type][1].item() for elm in DATA_SEED_LIST]
            kernel_type = kernel_type.replace('Additive', 'Add')
            kernel_type = kernel_type.replace('Product', 'Prod')
            kernel_type = kernel_type.replace('Modulated', 'Mod')
            kernel_type = kernel_type.replace('Diffusion', 'Dif')
            kernel_type = kernel_type.replace('Laplacian', 'Lap')
            kernel_type_str_list.append(kernel_type)
            nll_str_list.append('%6.4f(%6.4f)' % (np.mean(test_nll_list), np.std(test_nll_list) / len(test_nll_list) ** 0.5))
            rmse_str_list.append('%6.4f(%6.4f)' % (np.mean(test_rmse_list), np.std(test_rmse_list) / len(test_rmse_list) ** 0.5))
        print('/'.join(kernel_type_str_list))
        print('/'.join(nll_str_list))
        print('/'.join(rmse_str_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP with Mixed Variable Kernels')
    parser.add_argument('--data_type', dest='data_type', type=str, default=None, help='')
    parser.add_argument('--report', dest='report', default=False, action='store_true')

    args = parser.parse_args()
    if args.report:
        read_data()
        exit(0)

    print(args.data_type)
    print(DATA_SEED_LIST)
    print(KERNEL_TYPE_LIST)

    result_dict_dict = dict()

    for data_seed in DATA_SEED_LIST:
        train_data, test_data, nominal_num_cat = load_uci_data(data_type=args.data_type, random_seed=data_seed)

        result_dict = dict()
        for kernel_type in KERNEL_TYPE_LIST:

            train_x = torch.from_numpy(train_data[0])
            train_y = torch.from_numpy(train_data[1])
            test_x = torch.from_numpy(test_data[0])
            test_y = torch.from_numpy(test_data[1])

            train_y = train_y.squeeze(1)
            test_y = test_y.squeeze(1)
            gt_line = np.linspace(torch.min(test_y).item(),  torch.max(test_y).item(), 100)

            n_continuous = train_x.size(1) - len(nominal_num_cat)
            fourier_freq = []
            fourier_basis = []
            for ncat in nominal_num_cat:
                _, freq, basis = complete_graph_fourier(ncat)
                fourier_freq.append(freq)
                fourier_basis.append(basis)

            model = MixedVariableGPOptimize(
                n_continuous=n_continuous, fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                kernel_type=kernel_type)

            model.optimize(train_x=train_x, train_y=train_y)
            print('%5s(Train:%4d/Test:%4d) seed:%2d' % (args.data_type, train_x.shape[0], test_x.shape[0], data_seed))

            model.gp_model.eval()
            pred_distribution = model.gp_model(test_x)
            pred_mean = pred_distribution.loc.detach()
            pred_var = pred_distribution.variance.detach()

            test_nll = torch.mean(0.5 / pred_var * (test_y - pred_mean) ** 2 + 0.5 * torch.log(2 * math.pi * pred_var))
            test_rmse = torch.mean((test_y - pred_mean) ** 2) ** 0.5
            plt.plot(gt_line, gt_line, 'k-')
            plt.errorbar(test_y.numpy(), pred_mean.numpy(),
                         yerr=pred_var.clamp(min=0).numpy() ** 0.5, fmt='d')
            plt.title('%s[%d]_%s' % (args.data_type, data_seed, model.__class__.__name__), fontsize=8)
            filename = '%s_%s_%s.png' % (args.data_type, model.__class__.__name__, datetime.now().strftime('%H:%M:%S'))
            plt.savefig(os.path.join(os.path.split(__file__)[0], 'plots', filename))

            train_y_mean = torch.mean(train_y)
            print('MSE by train output mean : %7.4f' % torch.mean((test_y - train_y_mean) ** 2).item())
            result_dict[kernel_type] = (test_nll, test_rmse)

        result_dict_dict[data_seed] = result_dict

    for data_seed, result_dict in result_dict_dict.items():
        print('=' * 100)
        print('Data Seed : %2d' % data_seed)
        print('test nll')
        print('\n'.join(['%+.6E' % result_dict[kernel_type][0] for kernel_type in KERNEL_TYPE_LIST]))
        print('test MSE')
        print('\n'.join(['%7.4f' % result_dict[kernel_type][1] for kernel_type in KERNEL_TYPE_LIST]))

    torch.save(obj=result_dict_dict, f='%s.pth' % args.data_type)
