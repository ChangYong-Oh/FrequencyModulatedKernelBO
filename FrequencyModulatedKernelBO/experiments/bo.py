import os
import sys
import git
import time
import pathlib
from datetime import datetime

from typing import Tuple, List, Dict, Union, Optional

import torch
from torch import Tensor

import gpytorch

from FrequencyModulatedKernelBO.experiments.gp_models import MixedVariableGPBase, MixedVariableGPMCMC, MixedVariableGPOptimize,\
    MCMC_NUM_SAMPLES, MCMC_WARMUP_STEPS
from FrequencyModulatedKernelBO.acquisition.functions import AcquisitionFunction, AverageAcquisitionFunction, \
    expected_improvement_mean_sigma, optimization_as_estimation, m_hat_estimate, gram_cholesky_lower_inv
from FrequencyModulatedKernelBO.acquisition.optimization import initial_points_for_optimization, \
    optimize_hill_climbing_with_optimized_continuous, optimize_alternate
from FrequencyModulatedKernelBO.experiments.data.blackbox_evaluator import BlackboxEvaluator
from FrequencyModulatedKernelBO.experiments.gp_models import KERNEL_TYPE_LIST, SAMPLER_TYPE_LIST
from FrequencyModulatedKernelBO.experiments.utils import save_bo_exp, load_bo_exp, generate_exp_tag, last_file_in_directory
from FrequencyModulatedKernelBO.experiments.config import exp_dir_root, FILENAME_ZFILL_SIZE


def print_bo_exp(gp_model_info_str: str, blackbox_evaluator_info_str: str,
                 wallclock_time: List[datetime], eval_x: Tensor, eval_y: Tensor, minimize: bool) -> str:
    # Due to the separation of BO and evaluation, eval_x.size(0) - 1 == eval_y.size(0) is possible
    n_x, n_dim = eval_x.size()
    n_y = eval_y.numel()
    torch_opt = torch.min if minimize else torch.max
    torch_argopt = torch.argmin if minimize else torch.argmax
    is_discrete = [torch.all(eval_x[:, i] == eval_x[:, i].long()) for i in range(n_dim)]

    best_eval = torch.tensor([torch_opt(eval_y[:i + 1]).item() for i in range(n_y)])
    best_eval_ind = torch.tensor([torch_argopt(eval_y[:i + 1]).item() + 1 for i in range(n_y)])

    format_str = "%s %s-th / y = %+16.8f / best: %+16.8f at %s / x = %s"
    print_str = '>' * 80 + '\n'
    git_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../'))
    try:
        print_str += git.cmd.Git(git_dir).log(-1) + '\n'
    except OSError as err:
        print('OSError occured at git.cmd.Git, %s' % err)
    print_str += '-' * 80 + '\n'

    for i in range(n_y):
        print_str += format_str % (
            wallclock_time[i].strftime('%H:%M:%S'),
            str(i + 1).zfill(FILENAME_ZFILL_SIZE), eval_y[i].item(), best_eval[i].item(),
            str(best_eval_ind[i].item()).zfill(FILENAME_ZFILL_SIZE),
            ', '.join([('%2d' if is_discrete[j] else '%+8.4f') % eval_x[i, j].item() for j in range(n_dim)])
        )
        print_str += '\n'
    if n_x == n_y + 1:
        i = n_y
        print_str += "%s %s-th / y = ---------------- / best: ---------------- at %s / x = %s" % (
            '--:--:--', str(i + 1).zfill(FILENAME_ZFILL_SIZE), str(0) * FILENAME_ZFILL_SIZE,
            ', '.join([('%2d' if is_discrete[j] else '%+8.4f') % eval_x[i, j].item() for j in range(n_dim)])
        )
        print_str += '\n'

    # prev_best_eval_ind = best_eval_ind[-2] - 1
    # prev_best_x_str_list = [('%2d' if is_discrete[i] else '%+.4f') % eval_x[prev_best_eval_ind, i].item()
    #                         for i in range(n_dim)]
    # print_str += '      Previous best x (%3d-th): %s\n' % (prev_best_eval_ind + 1, ', '.join(prev_best_x_str_list))
    # n_prev_print = 50
    # for prev_ind in range(max(0, n_x - n_prev_print), n_x):
    #     prev_x_str_list = [('%2d' if is_discrete[i] else '%+.4f') % eval_x[prev_ind, i].item() for i in range(n_dim)]
    #     print_str += '          Suggested x (%3d-th): %s\n' % (prev_ind + 1, ', '.join(prev_x_str_list))

    print_str += 'Blackbox function : %s\n' % blackbox_evaluator_info_str
    print_str += 'Model : %s\n' % gp_model_info_str
    print_str += '<' * 80 + '\n'
    return print_str


def normalize_data(eval_x: torch.Tensor, eval_y: torch.Tensor,
                   optimized_features: Dict[int, Union[Tuple[int, int], Tensor]]):
    ndim = eval_x.size(1)
    normalization_mean = torch.zeros(ndim)
    normalization_std = torch.ones(ndim)
    normalized_input_info = dict()

    for k, v in optimized_features.items():
        if isinstance(v, Tensor):
            normalized_input_info[k] = v.clone()
        elif isinstance(v, tuple):
            normalization_mean[k] = torch.mean(eval_x[:, k], dim=0)
            normalization_std[k] = torch.std(eval_x[:, k], dim=0)
            normalized_input_info[k] = ((v[0] - normalization_mean[k]) / normalization_std[k],
                                        (v[1] - normalization_mean[k]) / normalization_std[k])
        else:
            raise NotImplementedError

    normalization_mean = normalization_mean.view(1, normalization_mean.numel())
    normalization_std = normalization_std.view(1, normalization_mean.numel())

    normalized_eval_x = (eval_x - normalization_mean) / normalization_std
    normalized_eval_y = (eval_y - torch.mean(eval_y)) / torch.std(eval_y)

    data = dict()
    data['normalized_eval_x'] = normalized_eval_x
    data['normalized_eval_y'] = normalized_eval_y
    data['eval_x_normalization_mean'] = normalization_mean
    data['eval_x_normalization_std'] = normalization_std
    data['normalized_input_info'] = normalized_input_info

    return data


def suggest(gp_model: MixedVariableGPBase, acq_type: str, optimized_features: Dict,
            eval_x: Tensor, eval_y: Tensor, use_log_y: bool = False) -> Tensor:
    assert acq_type.lower() in ['ei', 'est']
    # Minimization
    minimize = True
    torch_opt = torch.min if minimize else torch.max

    normarlized_data_dict = normalize_data(eval_x=eval_x, eval_y=torch.log(eval_y) if use_log_y else eval_y,
                                           optimized_features=optimized_features)
    normalized_eval_x = normarlized_data_dict['normalized_eval_x']
    normalized_eval_y = normarlized_data_dict['normalized_eval_y']
    eval_x_normalization_mean = normarlized_data_dict['eval_x_normalization_mean']
    eval_x_normalization_std = normarlized_data_dict['eval_x_normalization_std']
    normalized_input_info = normarlized_data_dict['normalized_input_info']

    if isinstance(gp_model, MixedVariableGPMCMC):
        while True:
            try:
                gp_model.sample(train_x=normalized_eval_x, train_y=normalized_eval_y)
                break
            except (gpytorch.utils.errors.NanError, RuntimeError):
                print('gpytorch.utils.errors.NanError occured.')
                pass
        if acq_type.lower() == 'ei':
            acq_function = AverageAcquisitionFunction(
                surrogate=gp_model, acq_function=expected_improvement_mean_sigma,
                data_x=normalized_eval_x, data_y=normalized_eval_y, minimize=minimize)
            incumbent = torch_opt(normalized_eval_y).item()
        elif acq_type.lower() == 'est':
            acq_function = AverageAcquisitionFunction(
                surrogate=gp_model, acq_function=optimization_as_estimation,
                data_x=normalized_eval_x, data_y=normalized_eval_y, minimize=minimize)
            raise NotImplementedError
            cholesky_lower_inv = gram_cholesky_lower_inv(gp_model=gp_model.gp_model, data_x=normalized_eval_x)
            incumbent = m_hat_estimate(
                gp_model=gp_model.gp_model, data_x=normalized_eval_x, data_y=normalized_eval_y,
                normalized_input_info=normalized_input_info,
                mean_train=acq_function.mean_train, cholesky_lower_inv=cholesky_lower_inv, maximize=not minimize)
    elif isinstance(gp_model, MixedVariableGPOptimize):
        gp_model.optimize(train_x=normalized_eval_x, train_y=normalized_eval_y)
        if acq_type.lower() == 'ei':
            acq_function = AcquisitionFunction(surrogate=gp_model, acq_function=expected_improvement_mean_sigma,
                                               data_x=normalized_eval_x, data_y=normalized_eval_y, minimize=minimize)
            incumbent = torch_opt(normalized_eval_y).item()
        elif acq_type.lower() == 'est':
            acq_function = AcquisitionFunction(surrogate=gp_model, acq_function=optimization_as_estimation,
                                               data_x=normalized_eval_x, data_y=normalized_eval_y, minimize=minimize)
            cholesky_lower_inv = gram_cholesky_lower_inv(gp_model=gp_model.gp_model, data_x=normalized_eval_x)
            incumbent = m_hat_estimate(
                gp_model=gp_model.gp_model, data_x=normalized_eval_x, data_y=normalized_eval_y,
                normalized_input_info=normalized_input_info,
                mean_train=acq_function.mean_train, cholesky_lower_inv=cholesky_lower_inv, maximize=not minimize)
    else:
        raise NotImplementedError
    acq_function.acq_func_kwargs = {'incumbent': incumbent}

    input_info = normarlized_data_dict['normalized_input_info']
    initial_points, points_to_avoid = initial_points_for_optimization(
        acquisition_function=acq_function, input_info=input_info, data_x=normalized_eval_x)
    normalized_new_x, best_acq = optimize_alternate(acquisition_function=acq_function, input_info=input_info,
                                                    initial_points=initial_points, points_to_avoid=points_to_avoid)
    print('The optimized acquisition value %+.6E' % best_acq.item())

    new_x = normalized_new_x * eval_x_normalization_std + eval_x_normalization_mean

    return new_x


def run_suggest(gp_model: MixedVariableGPBase, acq_type: str, blackbox_evaluator: BlackboxEvaluator, minimize: bool,
                eval_x: Tensor, eval_y: Tensor, use_log_y: bool, time_BO: Tensor, time_eval: Tensor,
                wallclock_time: List[datetime], save_file_prefix: str):
    assert acq_type.lower() in ['ei', 'est']
    assert eval_x.size(0) == time_BO.size(0)
    assert eval_y.size(0) == time_eval.size(0) == len(wallclock_time)
    assert eval_x.size(0) == eval_y.size(0)

    print('[%s]%3d-th suggestion began' % (datetime.now().strftime('%H:%M:%S'), eval_x.size(0) + 1))

    datetime_BO_begin = datetime.now()
    new_x = suggest(gp_model=gp_model, acq_type=acq_type, optimized_features=blackbox_evaluator.optimized_features,
                    eval_x=eval_x, eval_y=eval_y)
    datetime_BO_end = datetime.now()
    eval_x = torch.cat([eval_x, new_x])
    time_BO = torch.cat([time_BO, (datetime_BO_begin - datetime_BO_end).total_seconds() * torch.ones(1)])

    save_file = '_'.join([save_file_prefix, str(eval_x.size(0)).zfill(FILENAME_ZFILL_SIZE)])
    save_data_dict = {'eval_x': eval_x, 'eval_y': eval_y, 'use_log_y': use_log_y,
                      'time_BO': time_BO, 'time_eval': time_eval, 'wallclock_time': wallclock_time,
                      'minimize': minimize}
    save_bo_exp(save_file=save_file, gp_model=gp_model, acq_type=acq_type, blackbox_evaluator=blackbox_evaluator,
                save_data_dict=save_data_dict,
                print_data_dict={'print_func': print_bo_exp,
                                 'gp_model_info_str': gp_model.info_str,
                                 'blackbox_evaluator_info_str': blackbox_evaluator.info_str,
                                 'wallclock_time': wallclock_time, 'eval_x': eval_x, 'eval_y': eval_y,
                                 'minimize': minimize})

    print('[%s]%3d-th suggestion Finishes' % (datetime.now().strftime('%H:%M:%S'), eval_x.size(0)))
    print(pathlib.Path(save_file_prefix).resolve().parent)


def run_evaluate(gp_model: MixedVariableGPBase, acq_type: str, blackbox_evaluator: BlackboxEvaluator, minimize: bool,
                 eval_x: Tensor, eval_y: Tensor, use_log_y: bool, time_BO: Tensor, time_eval: Tensor, wallclock_time: List[datetime],
                 save_file_prefix: str):
    assert acq_type.lower() in ['ei', 'est']
    assert eval_x.size(0) == time_BO.size(0)
    assert eval_y.size(0) == time_eval.size(0) == len(wallclock_time)
    assert eval_x.size(0) - eval_y.size(0) == 1

    datatime_eval_begin = datetime.now()
    new_y = blackbox_evaluator(eval_x[-1:])
    datetime_eval_end = datetime.now()
    eval_y = torch.cat([eval_y, new_y.view(1, 1)], dim=0)

    time_eval = torch.cat([time_eval, (datatime_eval_begin - datetime_eval_end).total_seconds() * torch.ones(1)])
    wallclock_time.append(datetime_eval_end)

    save_data_dict = {'eval_x': eval_x, 'eval_y': eval_y, 'use_log_y': use_log_y,
                      'time_BO': time_BO, 'time_eval': time_eval, 'wallclock_time': wallclock_time,
                      'minimize': minimize}
    save_file = '_'.join([save_file_prefix, str(eval_x.size(0)).zfill(FILENAME_ZFILL_SIZE)])
    save_bo_exp(save_file=save_file, gp_model=gp_model, acq_type=acq_type, blackbox_evaluator=blackbox_evaluator,
                save_data_dict=save_data_dict,
                print_data_dict={'print_func': print_bo_exp,
                                 'gp_model_info_str': gp_model.info_str,
                                 'blackbox_evaluator_info_str': blackbox_evaluator.info_str,
                                 'wallclock_time': wallclock_time, 'eval_x': eval_x, 'eval_y': eval_y,
                                 'minimize': minimize},
                display=False)
    print(pathlib.Path(save_file_prefix).resolve().parent)


def init_bo(save_file: str, model_type: str, acq_type: str, use_log_y: bool, sampler_type: Optional[str] = None, minimize: bool = True,
            exp_dir: str = exp_dir_root()) -> str:
    assert acq_type.lower() in ['ei', 'est']
    assert model_type in KERNEL_TYPE_LIST
    assert model_type != 'NoContext'
    if sampler_type is not None:
        assert sampler_type in SAMPLER_TYPE_LIST

    exp_id = int(os.path.splitext(os.path.basename(os.path.normpath(save_file)))[0].split('_')[-1][-2:-1])
    _, _, blackbox_evaluator, BO_data = load_bo_exp(save_file)
    eval_x = BO_data['eval_x']
    eval_y = BO_data['eval_y']
    time_BO = BO_data['time_BO']
    time_eval = BO_data['time_eval']
    wallclock_time = BO_data['wallclock_time']

    optimized_features = blackbox_evaluator.optimized_features
    continuous_feature_dims = sorted([k for k, v in optimized_features.items() if isinstance(v, tuple)])
    discrete_feature_dims = sorted([k for k, v in optimized_features.items() if isinstance(v, Tensor)])
    n_continuous_features = len(continuous_feature_dims)
    n_dicrete_feature = len(discrete_feature_dims)
    assert set(continuous_feature_dims) == set(range(n_continuous_features))
    assert set(discrete_feature_dims) == set(range(n_continuous_features, n_continuous_features + n_dicrete_feature))
    n_continuous = n_continuous_features
    fourier_freq = [blackbox_evaluator.fourier_frequency[d] for d in discrete_feature_dims]
    fourier_basis = [blackbox_evaluator.fourier_basis[d] for d in discrete_feature_dims]

    normarlized_data_dict = normalize_data(eval_x=eval_x, eval_y=eval_y, optimized_features=optimized_features)
    normalized_eval_x = normarlized_data_dict['normalized_eval_x']
    normalized_eval_y = normarlized_data_dict['normalized_eval_y']

    if sampler_type is not None:
        gp_model = MixedVariableGPMCMC(n_continuous=n_continuous,
                                       fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                                       kernel_type=model_type, sampler_type=sampler_type,
                                       num_samples=MCMC_NUM_SAMPLES, warmup_steps=MCMC_WARMUP_STEPS)
        nan_error_cnt = 0
        while True:
            try:
                gp_model.sample(train_x=normalized_eval_x, train_y=normalized_eval_y)
                break
            except (gpytorch.utils.errors.NanError, RuntimeError):
                nan_error_cnt += 1
                print('gpytorch.utils.errors.NanError occured %d time(s).' % nan_error_cnt)
                pass
    else:
        gp_model = MixedVariableGPOptimize(n_continuous=n_continuous,
                                           fourier_freq=fourier_freq, fourier_basis=fourier_basis,
                                           kernel_type=model_type)

    exp_tag = generate_exp_tag(gp_model=gp_model, acq_type=acq_type,
                               blackbox_evaluator=blackbox_evaluator, exp_id=exp_id)
    save_dir = os.path.join(exp_dir, exp_tag)
    os.makedirs(save_dir)
    save_file_prefix = os.path.join(save_dir, exp_tag)

    save_file = '_'.join([save_file_prefix, str(eval_x.size(0)).zfill(FILENAME_ZFILL_SIZE)])
    save_bo_exp(save_file=save_file, gp_model=gp_model, acq_type=acq_type, blackbox_evaluator=blackbox_evaluator,
                save_data_dict={'eval_x': eval_x, 'eval_y': eval_y, 'use_log_y': use_log_y,
                                'time_BO': time_BO, 'time_eval': time_eval, 'wallclock_time': wallclock_time,
                                'minimize': minimize},
                print_data_dict={'print_func': print_bo_exp,
                                 'gp_model_info_str': gp_model.info_str,
                                 'blackbox_evaluator_info_str': blackbox_evaluator.info_str,
                                 'wallclock_time': wallclock_time, 'eval_x': eval_x, 'eval_y': eval_y,
                                 'minimize': minimize})

    return save_dir


def continue_bo(save_dir: str, max_eval: int, do_suggest: bool, do_evaluate: bool):
    if do_suggest and not do_evaluate:
        waiting_msg = 'Waiting for an evaluation to be completed.'
    elif not do_suggest and do_evaluate:
        waiting_msg = 'Waiting for a suggestion to be completed.'
    elif do_suggest and do_evaluate:
        waiting_msg = 'Suggestion and evaluation are running alternatively.'
    else:
        raise NotImplementedError
    sys.stdout.write(waiting_msg)
    while True:
        last_file, last_file_prefix = last_file_in_directory(save_dir)
        try:
            gp_model, acq_type, blackbox_evaluator, BO_data = load_bo_exp(last_file)
        except EOFError:
            print('EOFError occured.')
            continue
        assert 0 <= BO_data['eval_x'].size(0) - BO_data['eval_y'].size(0) <= 1
        if BO_data['eval_y'].size(0) >= max_eval:
            break
        if do_suggest and BO_data['eval_x'].size(0) == BO_data['eval_y'].size(0):
            sys.stdout.write('\n')
            run_suggest(gp_model=gp_model, acq_type=acq_type, blackbox_evaluator=blackbox_evaluator,
                        save_file_prefix=last_file_prefix, **BO_data)
            if not do_evaluate:
                sys.stdout.write(waiting_msg)
                sys.stdout.flush()
        if do_evaluate and BO_data['eval_x'].size(0) - BO_data['eval_y'].size(0) == 1:
            sys.stdout.write('\n')
            run_evaluate(gp_model=gp_model, acq_type=acq_type, blackbox_evaluator=blackbox_evaluator,
                         save_file_prefix=last_file_prefix, **BO_data)
            if not do_suggest:
                sys.stdout.write(waiting_msg)
                sys.stdout.flush()
        if do_suggest != do_evaluate:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(10)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return save_dir
