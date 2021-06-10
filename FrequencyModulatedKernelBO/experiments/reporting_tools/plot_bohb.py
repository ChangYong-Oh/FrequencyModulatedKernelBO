

import os
import json
import pickle

import numpy as np
import pandas as pd

import hpbandster.core.result as hpres


def read_recent_bohb_result(dirname):
    run_history = hpres.logged_results_to_HBS_result(dirname)

    all_runs = run_history.get_all_runs()
    print(len(all_runs))
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

    return pd.DataFrame(data=optimum_history[:, 1], index=optimum_history[:, 0].astype(np.int))

