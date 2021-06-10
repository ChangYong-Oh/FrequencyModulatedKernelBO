import os

import numpy as np
import matplotlib.pyplot as plt

from FrequencyModulatedKernelBO.experiments.config import exp_dir_root
from FrequencyModulatedKernelBO.experiments.gp_models import KERNEL_TYPE_LIST, SAMPLER_TYPE_LIST
from FrequencyModulatedKernelBO.experiments.data.exp_nas_nasnet import N_INIT_NAS_NASNET
from FrequencyModulatedKernelBO.experiments.reporting_tools.utils import read_GP_data, COLOR_DICT, MODEL_LIST



import os

import numpy as np
import pandas as pd

import hpbandster.core.result as hpres


def read_recent_bohb_result(dirname):
    run_history = hpres.logged_results_to_HBS_result(dirname)

    all_runs = run_history.get_all_runs()
    for i, run in enumerate(all_runs):
        if run.error_logs is not None:
            print(run)

    all_runs_dict = dict()
    for elm in all_runs:
        n_iter = elm.config_id[0]
        budget = elm.budget
        if n_iter in all_runs_dict:
            if elm.config_id in all_runs_dict[n_iter]:
                all_runs_dict[n_iter][elm.config_id][budget] = elm
            else:
                all_runs_dict[n_iter][elm.config_id] = {budget: elm}
        else:
            all_runs_dict[n_iter] = {elm.config_id: {budget: elm}}

    optimum_history = np.zeros((len(all_runs_dict), 2))
    for iter_num, runs_in_iter in all_runs_dict.items():
        budgets_in_iter = []
        min_losses_in_iter = []
        for config_id, run_data in runs_in_iter.items():
            min_losses_in_iter.append(min([value['loss'] for key, value in run_data.items()]))
            budgets_in_iter.append(sum([round(value['budget']) for key, value in run_data.items()]))
        optimum_history[iter_num, 0] = sum(budgets_in_iter)
        optimum_history[iter_num, 1] = min(min_losses_in_iter)

    optimum_history[:, 0] = np.cumsum(optimum_history[:, 0])
    optimum_history[:, 1] = np.exp(optimum_history[:, 1])

    return optimum_history


def merge_runs(all_runs: pd.DataFrame, run):
    all_runs = pd.concat([all_runs, pd.DataFrame(data=np.minimum.accumulate(run[:, 1]), index=run[:, 0].astype(np.int))], axis=1)
    for c in range(all_runs.shape[1]):
        r = 1
        while r < all_runs.shape[0] - 1:
            if np.isnan(all_runs.iloc[r, c]):
                r0 = r - 1
                r1 = r + 1
                while r1 < all_runs.shape[0] and np.isnan(all_runs.iloc[r1, c]):
                    r1 += 1
                if r1 < all_runs.shape[0]:
                    for r_mid in range(r0 + 1, r1):
                        x0 = all_runs.index[r0]
                        x1 = all_runs.index[r1]
                        x = all_runs.index[r_mid]
                        y0 = all_runs.iloc[r0, c]
                        y1 = all_runs.iloc[r1, c]
                        all_runs.iloc[r_mid, c] = (y1 - y0) / (x1 - x0) * (x - x0) + y0
                r = r1
            else:
                r += 1
    return all_runs


def bohb_mean_std(exp_dir: str) -> pd.DataFrame:
    all_runs = None
    for elm in os.listdir(exp_dir):
        dirname = os.path.join(exp_dir, elm)
        if 'BOHB_NASNET_FashionMNIST_[R' in elm and os.path.isdir(dirname):
            if all_runs is None:
                run = read_recent_bohb_result(dirname)
                all_runs = pd.DataFrame(data=np.minimum.accumulate(run[:, 1]), index=run[:, 0].astype(np.int))
            else:
                all_runs = merge_runs(all_runs=all_runs, run=read_recent_bohb_result(dirname))
    all_runs = all_runs.loc[~np.any(np.isnan(all_runs), axis=1), :]
    return all_runs


def sort_GP_dirname_NASNet(root_dirname):
    dirname_group_dict = dict()
    for dirname in os.listdir(root_dirname):
        if 'NASNASNET' == dirname[:9] and 'MV-K' in dirname and os.path.isdir(os.path.join(root_dirname, dirname)):
            model_info_str = [elm for elm in dirname.split('_') if 'MV-K' in elm][0]
            kernel_type = model_info_str.split('-')[1][1:]
            sampler_type = model_info_str.split('-')[2]
            assert kernel_type in KERNEL_TYPE_LIST
            tuple_key = (kernel_type, sampler_type)
            try:
                dirname_group_dict[tuple_key].append(dirname)
            except KeyError:
                dirname_group_dict[tuple_key] = [dirname]
        elif 'NASNASNET' == dirname[:9] and 'CTX-K' in dirname and os.path.isdir(os.path.join(root_dirname, dirname)):
            model_info_str = [elm for elm in dirname.split('_') if 'CTX-K' in elm][0]
            kernel_type = model_info_str.split('-')[1][1:]
            sampler_type = model_info_str.split('-')[2][1:]
            assert kernel_type in KERNEL_TYPE_LIST
            assert sampler_type in SAMPLER_TYPE_LIST
            tuple_key = (kernel_type, sampler_type)
            try:
                dirname_group_dict[tuple_key].append(dirname)
            except KeyError:
                dirname_group_dict[tuple_key] = [dirname]
        elif 'RegularizedEvolution' in dirname:
            tuple_key = ('RegularizedEvolution',)
            try:
                dirname_group_dict[tuple_key].append(dirname)
            except KeyError:
                dirname_group_dict[tuple_key] = [dirname]
    for k, v in dirname_group_dict.items():
        if len(v) != 4:
            print(('%s, %s' if len(k) == 2 else '%s') % k)
            for elm in v:
                print(elm)
    return dirname_group_dict


def summarize_GP_NASNet(root_dirname=exp_dir_root(), kernel_type=None, sampler_type=None):
    dirname_group_dict = sort_GP_dirname_NASNet(root_dirname)
    summary_dict = dict()
    for k, v in dirname_group_dict.items():
        if len(k) == 2:
            assert (kernel_type is None or k[0] == kernel_type) and (sampler_type is None or k[1] in sampler_type)
        elif len(k) == 1:
            assert k[0] == 'RegularizedEvolution'
        else:
            raise NotImplementedError
        all_eval_best_y = None
        for dirname in v:
            data_dict = read_GP_data(os.path.join(root_dirname, dirname))
            total_neval = data_dict['eval_y'].numel()
            # total_neval = 200 if 'KModulatedLaplacian-Optimize' in dirname else data_dict['eval_y'].numel()
            data_eval_y = data_dict['eval_y'].numpy().flatten()[:total_neval]
            total_eval_time = np.sum(-data_dict['time_eval'][N_INIT_NAS_NASNET:total_neval].numpy())
            total_acq_time = np.sum(
                    -data_dict['time_BO'][N_INIT_NAS_NASNET:total_neval].numpy()) if 'time_BO' in data_dict else 0
            print(dirname)
            print('%4d rounds without first %d init eval %10d seconds ( = %10d BO + %10d eval)' %
                  (data_eval_y.size, N_INIT_NAS_NASNET, total_eval_time + total_acq_time, total_acq_time, total_eval_time))
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


def mean_std_GP_NASNet(summary_dict):
    mean_std_dict = dict()
    for k, v in summary_dict.items():
        mean_std_dict[k] = {'mean': np.mean(v, axis=1), 'std': np.std(v, axis=1), 'n_runs': v.shape[1]}
    return mean_std_dict


def plot_summary_NASNet(max_eval, sampler_type, root_dirname=exp_dir_root()):
    mean_std_dict = mean_std_GP_NASNet(summarize_GP_NASNet(root_dirname, None, sampler_type))

    ordered_keys = []
    for m in MODEL_LIST + ['RegularizedEvolution']:
        for k in mean_std_dict.keys():
            if k[0] == m:
                ordered_keys.append(k)

    fig, ax = plt.subplots(1, 1)

    bohb_data = bohb_mean_std()
    mean = bohb_data.mean(axis=1).to_numpy()
    std = bohb_data.std(axis=1).to_numpy()
    n_runs = bohb_data.shape[1]
    stderr = std / n_runs ** 0.5
    plot_x = bohb_data.index.to_numpy() / 25
    # plot_ind = plot_x >= N_INIT_NAS_NASNET - 1
    # plot_x = plot_x[plot_ind]
    # mean = mean[plot_ind]
    # stderr = stderr[plot_ind]
    ax.plot(plot_x, mean, color='orange', label='BOHB', linestyle='-')
    ax.fill_between(plot_x, (mean - stderr), (mean + stderr), color='orange', alpha=0.1)
    for bohb_neval in range(-4, 0, 1):
        print('%s : \\num{%.3E}$\pm$\\num{%.4E}' % (('%s(%3d)' % ('BOHB', plot_x[bohb_neval])).center(12), mean[bohb_neval], stderr[bohb_neval]))

    for k in ordered_keys:
        v = mean_std_dict[k]
        if len(k) == 2:
            assert k[1] == sampler_type
        elif len(k) == 1:
            assert k[0] == 'RegularizedEvolution'
        else:
            raise NotImplementedError
        color = COLOR_DICT[k[0]]

        mean = v['mean'][N_INIT_NAS_NASNET - 1:]
        std = v['std'][N_INIT_NAS_NASNET - 1:]
        n_runs = v['n_runs']
        stderr = std / n_runs ** 0.5
        plot_x = list(np.arange(mean.size) + N_INIT_NAS_NASNET)

        label = k[0].replace('Laplacian', 'Lap')
        label = label.replace('Product', 'Prod')
        label = label.replace('Modulated', 'Mod')
        label = label.replace('RegularizedEvolution', 'RE')
        linestyle = '-'
        plot_x_max = max_eval - N_INIT_NAS_NASNET + 1
        ax.plot(plot_x[:plot_x_max], mean[:plot_x_max], color=color, label=label, linestyle=linestyle)
        ax.fill_between(plot_x[:plot_x_max],
                         (mean - stderr)[:plot_x_max], (mean + stderr)[:plot_x_max],
                         color=color, alpha=0.1)
        if label == 'RE':
            for re_neval in [200, 230, 400, 500]:
                print('%s : \\num{%.3E}$\pm$\\num{%.4E}' % (('%s(%3d)' % (label, re_neval)).center(12),
                                                            mean[re_neval - N_INIT_NAS_NASNET + 1],
                                                            stderr[re_neval - N_INIT_NAS_NASNET + 1]))
        n_eval = mean.size + N_INIT_NAS_NASNET - 1
        print('%s : \\num{%.3E}$\pm$\\num{%.4E}' % (('%s(%3d)' % (label, n_eval)).center(12), mean[-1], stderr[-1]))
    ax.legend(loc=1, ncol=1, fontsize=18)
    plt.tight_layout(h_pad=0, w_pad=0)
    ax.set_xlim((0, 600))
    # ax.axvline(x=200, linestyle='--', color='gray', alpha=0.5)
    # ax.legend(loc=9, ncol=3, fontsize=14)
    ax.legend(loc=1, ncol=1, fontsize=18)
    ax.set_ylim((None, 0.08))
    ax.set_xlabel('The number of evaluations', fontsize=10, labelpad=-1)
    ax.set_ylabel('Minimum', fontsize=10, labelpad=-1)
    # ax.set_xticks(np.arange(0, 601, 20), minor=True)
    # ax.set_yticks(np.arange(0.068, 0.079, 0.0005), minor=True)
    # ax.grid(which='both', linestyle=':')
    plt.show()

