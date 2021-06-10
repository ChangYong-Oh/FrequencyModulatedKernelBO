
from typing import Callable, Dict, Tuple, Union, List

import sys
import time
from tqdm import tqdm
from copy import deepcopy
from pathos import multiprocessing
import multiprocess.context as ctx

import torch
from torch import Tensor

import numpy as np
from scipy.optimize import minimize

from FrequencyModulatedKernelBO.acquisition.functions import AverageAcquisitionFunction

ctx._force_start_method('spawn')

REL_TOL = 1e-6
CONTINUOUS_SPRAY_STD = 0.001
CONTINUOUS_N_SPRAY = 50
CONTINUOUS_MAX_ITER = 1  # it seems that not many iteration is needed until convergence maybe less than 50
DISCRETE_MAX_ITER = 1
ACQ_OPTIM_N_CORES = 8


def f_discrete_fixed(acquisition_function: Callable, inds_continuous: List[int], initial_point: Tensor):
    """
    Discrete part of initial_point specify fixed values of discrete variables
    :param acquisition_function:
    :param inds_continuous:
    :param initial_point:
    :return:
    """
    if initial_point.ndim == 2:
        assert initial_point.size()[0] == 1
    else:
        initial_point = initial_point.view(1, -1)

    acquisition_function = deepcopy(acquisition_function)

    def f(x):
        x_continuous = torch.from_numpy(x).to(initial_point).view(1, len(inds_continuous)).\
            contiguous().requires_grad_(True)
        x_full = torch.clone(initial_point)
        x_full[0, inds_continuous] = x_continuous
        loss = -acquisition_function(x_full)
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        fval = loss.item()
        gradf = torch.autograd.grad(loss, x_continuous)[0].contiguous().view(-1)
        return fval, gradf.cpu().detach().contiguous().double().clone().numpy()

    return f


def scipy_optimize_wrapper(acquisition_function: AverageAcquisitionFunction, inds_continuous: List[int],
                           initial_point: Tensor, bounds: List, id: int):
    f = f_discrete_fixed(acquisition_function=acquisition_function, inds_continuous=inds_continuous,
                         initial_point=initial_point)
    x0 = initial_point[..., inds_continuous].numpy()
    result = minimize(fun=f, x0=x0, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': CONTINUOUS_MAX_ITER})
    return id, torch.from_numpy(result.x.astype(np.float32)), result.fun


def discrete_neighbors(points: Tensor, converged, input_info: Dict[int, Union[Tuple, Tensor]]):
    neighbors_list = []
    range_dict = dict()
    ind = 0
    inds_discrete = [k for k, v in input_info.items() if isinstance(v, Tensor)]
    for n in range(points.size()[0]):
        if not converged[n]:
            nth_point = points[n]  # size : ndim
            nth_point_nbd_list = []
            for d in inds_discrete:
                nth_point_d_dim_cand = input_info[d][nth_point[d].long()].nonzero(as_tuple=False).squeeze(1)
                nth_point_d_dim_nbds = nth_point.view(1, -1).repeat((nth_point_d_dim_cand.numel(), 1))
                nth_point_d_dim_nbds[:, d] = nth_point_d_dim_cand
                nth_point_nbd_list.append(nth_point_d_dim_nbds)
            neighbors_list.append(torch.cat(nth_point_nbd_list, dim=0))
            range_dict[n] = (ind, ind + neighbors_list[-1].size()[0])
            ind = range_dict[n][1]
        else:
            range_dict[n] = (ind, ind)

    # Making results size of 'n_neighbors X 1 X search_space_dim'
    return torch.cat(neighbors_list, dim=0), range_dict


def random_points(n: int, input_info: Dict[int, Union[Tuple, Tensor]], device):
    points = torch.zeros(n, len(input_info), device=device)
    for k, val in input_info.items():
        if isinstance(val, Tuple):
            points[:, k] = torch.rand_like(points[:, k]) * (val[1] - val[0]) + val[0]
        elif isinstance(val, Tensor):
            points[:, k] = torch.randint_like(points[:, k], high=val.size()[0])
    return points


def optimize_continuous_part_scipy_minimize(acquisition_function: AverageAcquisitionFunction,
                                            input_info: Dict[int, Union[Tuple, Tensor]],
                                            initial_points: Tensor) -> Tuple[Tensor, Tensor]:
    """Generate a set of candidates using `scipy.optimize.minimize`.

    Optimizes continuous variables of an acquisition function starting from a set of initial candidates
    using `scipy.optimize.minimize` via a numpy converter.

    Args:
        acquisition_function: Acquisition function to be used.
        input_info:
        initial_points: Starting points for optimization. 'n_candidates X q X ndim'

    Returns:
        2-element tuple containing

    """
    assert initial_points.ndim == 2
    inds_continuous = sorted([k for k, v in input_info.items() if isinstance(v, Tuple)])
    bounds = [(input_info[k][0].item(), input_info[k][1].item()) for k in inds_continuous]

    n_inits = initial_points.size()[0]
    optimized_points = initial_points.clone()
    optimized_negated_values = initial_points.new_zeros(n_inits)

    n_processes = max(multiprocessing.cpu_count() - 2, multiprocessing.cpu_count() // 2, 1)
    pool = multiprocessing.Pool(n_processes)
    args_list = []
    for n in range(n_inits):
        args_list.append((acquisition_function, inds_continuous, initial_points[n], bounds, n))

    instance_id, optimum_loc, optimum_val = list(zip(*pool.starmap_async(scipy_optimize_wrapper, args_list).get()))
    instance_id = torch.LongTensor(instance_id)
    optimum_loc = torch.stack(optimum_loc, dim=0)
    optimum_neg_val = torch.tensor(optimum_val)
    optimized_points[instance_id.view(-1, 1).repeat(1, len(inds_continuous)),
                     torch.LongTensor(inds_continuous).view(1, -1).repeat(instance_id.numel(), 1)] = optimum_loc
    optimized_negated_values[instance_id] = optimum_neg_val

    return optimized_points, optimized_negated_values


def optimize_hill_climbing_with_optimized_continuous(
        acquisition_function: AverageAcquisitionFunction,
        input_info: Dict[int, Union[Tuple, Tensor]],
        initial_points: Tensor, points_to_avoid: Tensor) -> Tuple[Tensor, Tensor]:

    inds_discrete = [k for k, v in input_info.items() if isinstance(v, Tensor)]
    points_to_avoid_discrete = points_to_avoid[:, inds_discrete].unsqueeze(0)

    converged = initial_points.new_zeros(initial_points.size()[0]).bool()
    optimum_loc = initial_points.clone()
    _, optimum_neg_val = optimize_continuous_part_scipy_minimize(
        acquisition_function=acquisition_function, input_info=input_info, initial_points=optimum_loc)
    start_time = time.time()
    print('Hill Climbing discrete variables while optimizing continuous variables for fixed discrete variables')
    while not torch.all(converged):
        print('-' * 50)
        # while fixing continuous part, enumerating all 1-hop neighbors
        neighbors, nbd_range_dict = discrete_neighbors(points=optimum_loc, converged=converged, input_info=input_info)
        # for each neighbors, optimizing continuous part while fixing discrete part
        nbd_optimized, nbd_neg_acq_val = optimize_continuous_part_scipy_minimize(
            acquisition_function=acquisition_function, input_info=input_info, initial_points=neighbors)
        # Handling discrete points to avoid
        inds_to_avoid = torch.any(
            torch.all(nbd_optimized[:, inds_discrete].unsqueeze(1) == points_to_avoid_discrete, dim=2), dim=1)
        nbd_neg_acq_val[inds_to_avoid] = float('inf')
        # with optimized continuous part, pick the best (min negated acq. val.) neighbor
        for i, (ind1, ind2) in nbd_range_dict.items():
            if ind1 != ind2:
                min_neg_val, ind_shift = torch.min(nbd_neg_acq_val[ind1:ind2], dim=0)
                if min_neg_val < optimum_neg_val[i]:
                    print('%4d-th acquisition value from %+.6f to %+.6f' %
                          (i, -optimum_neg_val[i].item(), -min_neg_val.item()))
                    optimum_neg_val[i] = min_neg_val
                    optimum_loc[i] = nbd_optimized[ind1 + ind_shift]
                else:
                    converged[i] = True
    sys.stdout.write('\n')
    print('Acquisition function optimization : %.2f seconds' % (time.time() - start_time))
    best_neg_acq, best_ind = torch.min(optimum_neg_val, dim=0)
    return optimum_loc[best_ind], -best_neg_acq


def best_acq_val_nbd(acquisition_function: AverageAcquisitionFunction,
                     input_info: Dict[int, Union[Tuple, Tensor]],
                     point: Tensor, points_to_avoid: Tensor,
                     inds_discrete: List[int]) -> Tuple[Tensor, float]:
    assert point.ndim == 1
    assert points_to_avoid.ndim == 2
    points_to_avoid_discrete = points_to_avoid[:, inds_discrete].unsqueeze(0)
    # while fixing continuous part, enumerating all 1-hop neighbors
    neighbors, _ = discrete_neighbors(points=point.view(1, -1),
                                      converged=point.new_zeros(1).bool(), input_info=input_info)
    nbd_acq_val = acquisition_function(neighbors).detach()
    # Handling discrete points to avoid
    inds_to_avoid = \
        torch.any(torch.all(neighbors[:, inds_discrete].unsqueeze(1) == points_to_avoid_discrete, dim=2), dim=1)
    nbd_acq_val[inds_to_avoid] = -float('inf')
    # with optimized continuous part, pick the best (min negated acq. val.) neighbor
    max_acq_val, max_acq_val_ind = torch.max(nbd_acq_val, dim=0)
    max_acq_val = max_acq_val.item()
    max_acq_val_nbd = neighbors[max_acq_val_ind]
    return max_acq_val_nbd, max_acq_val


def hill_climbing_for_given_continuous(
        acquisition_function: AverageAcquisitionFunction, input_info: Dict[int, Union[Tuple, Tensor]],
        initial_point: Tensor, points_to_avoid: Tensor,
        inds_discrete: List[int]) -> Tuple[Tensor, float]:
    assert initial_point.ndim == 1
    assert points_to_avoid.ndim == 2
    optimum_loc = initial_point
    optimum_acq_val = acquisition_function(initial_point.view(1, -1)).item()
    for _ in range(DISCRETE_MAX_ITER):
        max_acq_val_nbd, max_acq_val = best_acq_val_nbd(
            acquisition_function=acquisition_function, input_info=input_info, point=optimum_loc,
            points_to_avoid=points_to_avoid, inds_discrete=inds_discrete)
        if max_acq_val > optimum_acq_val:
             optimum_acq_val = max_acq_val
             optimum_loc = max_acq_val_nbd
        else:
            break
    return optimum_loc, optimum_acq_val


def scipy_minimize_for_given_discrete(acquisition_function: AverageAcquisitionFunction,
                                      initial_point: Tensor, inds_continuous: List[int],
                                      bounds_continuous: List[Tuple]) -> Tuple[Tensor, Tensor]:
    assert initial_point.ndim == 1
    _, optimum_loc_continuous, optimum_neg_acq_val = scipy_optimize_wrapper(
        acquisition_function=acquisition_function, inds_continuous=inds_continuous, initial_point=initial_point,
        bounds=bounds_continuous, id=0)
    optimum_loc = initial_point.clone()
    optimum_loc[inds_continuous] = optimum_loc_continuous
    return optimum_loc, -1.0 * optimum_neg_acq_val


def alternate_wrapper(acquisition_function: AverageAcquisitionFunction,
                      input_info: Dict[int, Union[Tuple, Tensor]],
                      initial_point: Tensor, points_to_avoid: Tensor, instance_id: int):
    assert initial_point.ndim == 1
    assert points_to_avoid.ndim == 2
    inds_continuous = sorted([k for k, v in input_info.items() if isinstance(v, Tuple)])
    inds_discrete = [k for k, v in input_info.items() if isinstance(v, Tensor)]
    bounds_continuous = [(input_info[k][0].item(), input_info[k][1].item()) for k in inds_continuous]
    loc = initial_point

    if torch.any(torch.all(loc[inds_discrete].view(1, -1) == points_to_avoid[:, inds_discrete], dim=1)):
        loc, acq_val = best_acq_val_nbd(acquisition_function=acquisition_function, input_info=input_info, point=loc,
                                        points_to_avoid=points_to_avoid, inds_discrete=inds_discrete)
    prev_loc = loc.clone()
    prev_acq_val = acquisition_function(prev_loc.view(1, -1)).item()
    init_acq_val = prev_acq_val

    decrease_count = 0
    for n in range(5000):
        loc_half, acq_val_half = hill_climbing_for_given_continuous(
            acquisition_function=acquisition_function, input_info=input_info, initial_point=prev_loc,
            points_to_avoid=points_to_avoid, inds_discrete=inds_discrete)
        loc, acq_val = scipy_minimize_for_given_discrete(
            acquisition_function=acquisition_function, initial_point=loc_half, inds_continuous=inds_continuous,
            bounds_continuous=bounds_continuous)
        large_enough_change = abs((prev_acq_val - acq_val) / max(abs(acq_val), 1.0)) > REL_TOL
        if acq_val < prev_acq_val:
            print('        Decreased from %+.6E to %+.6E' % (prev_acq_val, acq_val))
            sys.stdout.flush()
            decrease_count += 1
            if decrease_count > 9:
                print('%4d updates from %+.6E to %+.6E' % (n, init_acq_val, acq_val))
                return instance_id, prev_loc, prev_acq_val
        if not large_enough_change:
            # print('        Marginal change from %+.6E to %+.6E' % (prev_acq_val, acq_val))
            print('%4d updates from %+.6E to %+.6E' % (n, init_acq_val, acq_val))
            sys.stdout.flush()
            return instance_id, loc, acq_val
        prev_loc = loc.clone()
        prev_acq_val = acq_val

    print('%4d updates from %+.6E to %+.6E' % (n, init_acq_val, acq_val))
    sys.stdout.flush()

    return instance_id, loc, acq_val


def optimize_alternate(
        acquisition_function: AverageAcquisitionFunction,
        input_info: Dict[int, Union[Tuple, Tensor]],
        initial_points: Tensor, points_to_avoid: Tensor) -> Tuple[Tensor, Tensor]:
    assert initial_points.ndim == 2

    n_inits = initial_points.size(0)
    optimized_points = initial_points.clone()
    optimized_values = initial_points.new_zeros(n_inits)

    n_processes = max(multiprocessing.cpu_count() // ACQ_OPTIM_N_CORES, 1)
    pool = multiprocessing.Pool(n_processes)
    args_list = []
    for n in range(n_inits):
        args_list.append((acquisition_function, input_info, initial_points[n], points_to_avoid, n))

    start_time = time.time()
    print('Alternating continuous optimization (L-BFGS-B) and discrete optimization (Hill climbing)')
    print('For %d initial points with updates : ' % n_inits)
    sys.stdout.flush()
    instance_id, optimum_loc, optimum_val = list(zip(*pool.starmap_async(alternate_wrapper, args_list).get()))
    print('Acquisition function optimization : %.2f seconds' % (time.time() - start_time))
    sys.stdout.flush()
    instance_id = torch.LongTensor(instance_id)
    optimized_points[instance_id] = torch.stack(optimum_loc, dim=0)
    optimized_values[instance_id] = torch.tensor(optimum_val)

    best_acq, best_ind = torch.max(optimized_values, dim=0)
    mask_similar_val = torch.isclose(optimized_values, best_acq, rtol=1e-6)
    if torch.sum(mask_similar_val) > 1:
        optimized_points_sub = optimized_points[mask_similar_val]
        optimized_values_sub = optimized_values[mask_similar_val]
        unique_row_ind = \
            [i for i in range(optimized_points_sub.size(0))
             if torch.sum(torch.all(torch.isclose(optimized_points_sub[i], optimized_points_sub[i:]), dim=1)) == 1]
        rnd_ind = np.random.choice(unique_row_ind)
        return optimized_points_sub[rnd_ind], optimized_values_sub[rnd_ind]
    else:
        return optimized_points[best_ind], best_acq


def spray_points(data_x: Tensor, input_info: Dict[int, Union[Tuple, Tensor]]):
    inds_continuous = sorted([k for k, v in input_info.items() if isinstance(v, Tuple)])
    inds_discrete = sorted([k for k, v in input_info.items() if isinstance(v, Tensor)])
    lower_bounds = torch.stack([(input_info[k][0]) for k in inds_continuous])
    upper_bounds = torch.stack([(input_info[k][1]) for k in inds_continuous])
    data_x_spray = data_x.unsqueeze(1).repeat(1, CONTINUOUS_N_SPRAY, 1)
    data_x_spray[:, :, inds_continuous] += torch.randn_like(data_x_spray[:, :, inds_continuous]) * CONTINUOUS_SPRAY_STD
    data_x_spray = data_x_spray.view(-1, data_x_spray.size(-1))
    for i, ind in enumerate(inds_continuous):
        below_lower_bound_mask = data_x_spray[..., ind] < lower_bounds[i]
        above_upper_bound_mask = data_x_spray[..., ind] > upper_bounds[i]
        data_x_spray[below_lower_bound_mask, i] = 2 * lower_bounds[i] - data_x_spray[below_lower_bound_mask, i]
        data_x_spray[above_upper_bound_mask, i] = 2 * upper_bounds[i] - data_x_spray[above_upper_bound_mask, i]
        assert torch.all(torch.logical_and(lower_bounds[i] <= data_x_spray[..., ind],
                                           data_x_spray[..., ind] <= upper_bounds[i]))
    chosen_ind = np.random.choice(inds_discrete, size=(data_x_spray.size()[0], ), replace=True)
    for i, ind in enumerate(chosen_ind):
        data_x_spray[i, ind] = np.random.randint(0, input_info[ind].size()[0])
    return data_x_spray


def initial_points_for_optimization(acquisition_function: AverageAcquisitionFunction,
                                    input_info: Dict[int, Union[Tuple, Tensor]],
                                    data_x: Tensor) -> Tuple[Tensor, Tensor]:
    inds_continuous = sorted([k for k, v in input_info.items() if isinstance(v, Tuple)])
    lower_bounds = torch.stack([(input_info[k][0]) for k in inds_continuous])
    upper_bounds = torch.stack([(input_info[k][1]) for k in inds_continuous])

    n_data = data_x.size()[0]
    acq_value_on_data = acquisition_function(data_x)
    n_init_sub1 = 1  # points with topk acq_values
    n_random = 49  # uniformly random points
    acq_value_topk, acq_value_topk_ind = torch.topk(acq_value_on_data, k=min(2 * n_init_sub1, n_data))
    points_to_avoid = data_x[torch.arange(max(0, n_data - 1), n_data)]

    init_points_ind = acq_value_topk_ind[:n_init_sub1]
    init_points = torch.cat([data_x[init_points_ind],
                             random_points(n=n_random, input_info=input_info, device=data_x.device)], dim=0)
    n_continuous_perturb = CONTINUOUS_N_SPRAY
    init_points = init_points.unsqueeze(1).repeat(1, n_continuous_perturb, 1)
    init_points[:, :, inds_continuous] += torch.randn_like(init_points[:, :, inds_continuous]) * CONTINUOUS_SPRAY_STD
    init_points = init_points.view(-1, init_points.size(-1))
    for i, ind in enumerate(inds_continuous):
        below_lower_bound_mask = init_points[..., ind] < lower_bounds[i]
        above_upper_bound_mask = init_points[..., ind] > upper_bounds[i]
        init_points[below_lower_bound_mask, i] = 2 * lower_bounds[i] - init_points[below_lower_bound_mask, i]
        init_points[above_upper_bound_mask, i] = 2 * upper_bounds[i] - init_points[above_upper_bound_mask, i]
        assert torch.all(torch.logical_and(lower_bounds[i] <= init_points[..., ind],
                                           init_points[..., ind] <= upper_bounds[i]))

    start_time = time.time()

    optimized_init, optimized_neg_val = init_points, -1.0 * acquisition_function(init_points).detach()
    best_neg_acq_val, best_ind = torch.min(optimized_neg_val.view(-1, n_continuous_perturb), dim=-1)
    best_continuous_init = \
        optimized_init.view(-1, n_continuous_perturb, init_points.size(-1))[torch.arange(best_ind.numel()), best_ind]
    _, best_k_ind = torch.topk(best_neg_acq_val.view(-1), k=20, dim=-1, largest=False)
    best_init = best_continuous_init[best_k_ind]

    rnd_points = random_points(n=100000, input_info=input_info, device=data_x.device)
    rnd_neg_val = -1.0 * acquisition_function(rnd_points).detach()
    _, best_k_ind_rnd = torch.topk(rnd_neg_val.view(-1), k=20, dim=-1, largest=False)
    best_rnd = rnd_points[best_k_ind_rnd]

    best_init = torch.cat([best_init, best_rnd], dim=0)

    print('Acquisition function %d initial points selection from %d points : %.2f seconds'
          % (best_init.size()[0], init_points.size()[0] + rnd_points.size()[0], time.time() - start_time))

    return best_init, points_to_avoid
