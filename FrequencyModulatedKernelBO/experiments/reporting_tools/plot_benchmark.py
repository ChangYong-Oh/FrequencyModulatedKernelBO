import os
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FrequencyModulatedKernelBO.experiments.config import exp_dir_root
from FrequencyModulatedKernelBO.experiments.data.benchmark_functions import N_INIT_BENCHMARK
from FrequencyModulatedKernelBO.experiments.gp_models import KERNEL_TYPE_LIST
from FrequencyModulatedKernelBO.experiments.reporting_tools.utils import read_GP_data, COLOR_DICT, MODEL_LIST


N_RUN_SMAC = 5
N_RUN_TPE = 5


def sort_FM_dirname(root_dirname):
    dirname_group_dict = dict()
    for dirname in os.listdir(root_dirname):
        if len(dirname.split('_')) > 1 and dirname.split('_')[1].split('-')[0] == 'MV' \
                and os.path.isdir(os.path.join(root_dirname, dirname)):
            exp_type = dirname.split('_')[0]
            kernel_type = dirname.split('_')[1].split('-')[1][1:]
            assert kernel_type in KERNEL_TYPE_LIST
            tuple_key = (exp_type, kernel_type)
            try:
                dirname_group_dict[tuple_key].append(dirname)
            except KeyError:
                dirname_group_dict[tuple_key] = [dirname]
    for k, v in dirname_group_dict.items():
        if len(v) != 5:
            print('%s, %s' % k)
            for elm in v:
                print(elm)
    return dirname_group_dict


def summarize_FM(root_dirname=exp_dir_root(), exp_type=None, kernel_type=None):
    dirname_group_dict = sort_FM_dirname(root_dirname)
    summary_dict = dict()
    for k, v in dirname_group_dict.items():
        if (exp_type is None or k[0] == exp_type) and (kernel_type is None or k[1] == kernel_type):
            all_eval_best_y = None
            for dirname in v:
                data_dict = read_GP_data(os.path.join(root_dirname, dirname))
                data_eval_y = data_dict['eval_y'].numpy().flatten()
                data_best_y = np.zeros_like(data_eval_y)
                for i in range(data_best_y.size):
                    data_best_y[i] = np.min(data_eval_y[:i + 1])
                if all_eval_best_y is None:
                    all_eval_best_y = data_best_y.reshape((-1, 1))
                else:
                    n_eval = min(all_eval_best_y.shape[0], data_best_y.size)
                    all_eval_best_y = np.hstack((all_eval_best_y[:n_eval], data_best_y[:n_eval].reshape((-1, 1))))
            summary_dict[k] = all_eval_best_y
    return summary_dict


def summarize_cocabo(exp_type, root_dirname=exp_dir_root()):
    cocabo_dirname = [elm for elm in os.listdir(root_dirname) if ('%s_CoCaBO' % exp_type) == elm][0]
    filename_group_dict = dict()
    for filename in os.listdir(os.path.join(root_dirname, cocabo_dirname)):
        if re.search(u"_df_s[0-5]", filename) is not None:
            best_y = \
                -1.0 * pd.read_pickle(os.path.join(root_dirname, cocabo_dirname, filename))['best_value'].to_numpy()
            word_list = filename.split('_')
            key = (exp_type, 'CoCaBO-%.1f' % float(word_list[word_list.index('mix') + 1]))
            try:
                filename_group_dict[key] = np.hstack([filename_group_dict[key], best_y.reshape(-1, 1)])
            except KeyError:
                filename_group_dict[key] = best_y.reshape(-1, 1)
    data_dict = dict()
    for key, value in filename_group_dict.items():
        data_dict[key] = np.vstack([np.tile(filename_group_dict[key][:1], (N_INIT_BENCHMARK - 1, 1)),
                                    filename_group_dict[key]])
    return data_dict


def mean_std_from_summary(summary_dict):
    mean_std_dict = dict()
    for k, v in summary_dict.items():
        mean_std_dict[k] = {'mean': np.mean(v, axis=1), 'std': np.std(v, axis=1), 'n_runs': v.shape[1]}
    return mean_std_dict


def mean_std_SMAC(exp_type, root_dirname=exp_dir_root()):
    smac_file = [elm for elm in os.listdir(root_dirname) if exp_type in elm and 'smac' in elm]
    assert len(smac_file) == 1
    smac_file = os.path.join(root_dirname, smac_file[0])
    with open(smac_file, 'rb') as f:
        smac_data = pickle.load(f)
    smac_data['n_runs'] = N_RUN_SMAC
    return {(exp_type, 'smac'): smac_data}


def mean_std_TPE(exp_type, root_dirname=exp_dir_root()):
    tpe_file = [elm for elm in os.listdir(root_dirname) if exp_type in elm and 'tpe' in elm]
    assert len(tpe_file) == 1
    tpe_file = os.path.join(root_dirname, tpe_file[0])
    with open(tpe_file, 'rb') as f:
        tpe_data = pickle.load(f)
    tpe_data['n_runs'] = N_RUN_TPE
    return {(exp_type, 'tpe'): tpe_data}


def plot_summary_synthetic(exp_type, plot_type: str, root_dirname=exp_dir_root()):
    mean_std_dict = mean_std_from_summary(summarize_FM(root_dirname, exp_type, None))
    mean_std_dict.update(mean_std_from_summary(summarize_cocabo(exp_type, root_dirname)))
    mean_std_dict.update(mean_std_SMAC(exp_type, root_dirname))
    mean_std_dict.update(mean_std_TPE(exp_type, root_dirname))

    assert plot_type in ['main', 'main-legend', 'supp']

    # ordered_keys = []
    # for m in MODEL_LIST:
    #     for k in mean_std_dict.keys():
    #         if k[1] == m:
    #             ordered_keys.append(k)

    if plot_type == 'main-legend':
        method_names = ['smac', 'tpe',
                        'ModulatedDiffusion', 'ModulatedLaplacian',
                        'CoCaBO-0.0', 'CoCaBO-0.5', 'CoCaBO-1.0']
        legend_ncol = 7
    elif plot_type == 'main':
        method_names = ['smac', 'tpe', '',
                        'ModulatedDiffusion', 'ModulatedLaplacian', '',
                        'CoCaBO-0.0', 'CoCaBO-0.5', 'CoCaBO-1.0']
        legend_ncol = 3
    else:
        method_names = ['smac', 'tpe', '',
            'AdditiveDiffusion', 'ProductDiffusion', 'ModulatedDiffusion',
            'AdditiveLaplacian', 'ProductLaplacian', 'ModulatedLaplacian',
            'CoCaBO-0.0', 'CoCaBO-0.5', 'CoCaBO-1.0']
        legend_ncol = 4
    ordered_keys = [(exp_type, elm) for elm in method_names]
    y_min = min([min(mean_std_dict[k]['mean'][N_INIT_BENCHMARK - 1:]) for k in ordered_keys if k[1] != '' and k in mean_std_dict])
    y_max = max([max(mean_std_dict[k]['mean'][N_INIT_BENCHMARK - 1:]) for k in ordered_keys if k[1] != '' and k in mean_std_dict])

    fig, ax = plt.subplots(nrows=1, ncols=1)

    print('-' * 50)
    print('-' * 50)
    print('-' * 50)
    print(exp_type)
    print_str_list = []
    for k in ordered_keys:
        if len(k) == 2:
            assert k[0] == exp_type
        else:
            raise NotImplementedError
        if k[1] == '':
            plt.plot(100, (y_min + y_max) * 0.5, color='white', label=' ')
            continue

        if k not in mean_std_dict:
            continue

        v = mean_std_dict[k]
        color = COLOR_DICT[k[1]]

        if k[0] == 'XGBFashionMNIST':
            mean = v['mean'][N_INIT_BENCHMARK - 1:100]
            std = v['std'][N_INIT_BENCHMARK - 1:100]
        else:
            mean = v['mean'][N_INIT_BENCHMARK - 1:]
            std = v['std'][N_INIT_BENCHMARK - 1:]
        n_runs = v['n_runs']
        plot_x = list(np.arange(mean.size) + N_INIT_BENCHMARK)

        label = k[1].replace('Laplacian', 'Lap')
        label = label.replace('Diffusion', 'Dif')
        label = label.replace('Additive', 'Add')
        label = label.replace('Product', 'Prod')
        label = label.replace('Modulated', 'Mod')
        label = label.replace('Augmented', 'Aug')
        label = label.replace('smac', 'SMAC')
        label = label.replace('tpe', 'TPE')
        if 'Laplacian' in k[1]:
            linestyle = '-'
        elif 'CoCaBO' in k[1]:
            linestyle = '--'
        elif 'Diffusion' in k[1]:
            linestyle = ':'
        else:
            linestyle = '-.'
        if plot_type == 'main-legend':
            ax.plot(0, 0, color=color, label=label, linestyle=linestyle)
        else:
            ax.plot(plot_x, mean, color=color, label=label, linestyle=linestyle)
            ax.fill_between(plot_x, mean - std / n_runs ** 0.5, mean + std / n_runs ** 0.5, color=color, alpha=0.1)
        if plot_type == 'supp':
            print("%-18s & $%+9.4f\\pm%7.4f$ \\\\" % (label, mean[-1], std[-1] / n_runs ** 0.5))
        elif plot_type == 'main':
            print_str_list.append("$%+5.3f\\pm%5.3f$" % (mean[-1], std[-1] / n_runs ** 0.5))
    if plot_type == 'main-legend':
        fig.tight_layout(h_pad=0, w_pad=0, rect=[0, 0, 1, 0.8])
        ax.legend(loc=9, ncol=legend_ncol, fontsize=10, bbox_to_anchor=(0.45, 1.2))
        fig.set_size_inches(9.0, 4.0)
        plt.savefig('%s.png' % plot_type)
    elif plot_type == 'supp':
        fig.tight_layout(h_pad=0, w_pad=0)
        ax.legend(ncol=legend_ncol)
        # plt.title(exp_type)
        plt.savefig('%s_%s.png' % (exp_type, plot_type))
    else:
        print(' & '.join(print_str_list))
        fig.tight_layout(h_pad=0, w_pad=0)
        plt.savefig('%s_%s.png' % (exp_type, plot_type))
    plt.show()


if __name__ == '__main__':
    root_dirname_ = '/home/username/Experiments/FrequencyModulatedKernelBO'
    plot_summary_synthetic(exp_type='Func2C', plot_type='main-legend', root_dirname=root_dirname_)
    for exp_type_ in ['Func2C', 'Func3C', 'Ackley5C', 'SVMBoston', 'XGBFashionMNIST']:
        print(root_dirname_, exp_type_)
        plot_summary_synthetic(exp_type=exp_type_, plot_type='main', root_dirname=root_dirname_)
        plot_summary_synthetic(exp_type=exp_type_, plot_type='supp', root_dirname=root_dirname_)