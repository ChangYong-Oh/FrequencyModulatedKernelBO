# Modified from below
#==========================================
# Title:  syntheticFunctions.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

from typing import Optional

import os

import numpy as np

from sklearn.svm import NuSVR
from sklearn.datasets import load_boston, fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import xgboost as xgb

import torch

from FrequencyModulatedKernelBO.experiments.config import data_dir_root
from FrequencyModulatedKernelBO.experiments.data.blackbox_evaluator import BlackboxEvaluator

N_INIT_BENCHMARK = 10


# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
    X = np.asarray(X)
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300

# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html
# =============================================================================
def mysixhumpcamp(X):
    X = np.asarray(X)
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10

# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
    X = np.asarray(X) / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50


def func2C(ht_list, X):
    # ht is a categorical index
    # X is a continuous variable
    X = X * 2

    assert len(ht_list) == 2
    ht1 = ht_list[0]
    ht2 = ht_list[1]

    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:  # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:  # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:  # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)

    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
    return y.astype(float)


def func3C(ht_list, X):
    # ht is a categorical index
    # X is a continuous variable
    X = np.atleast_2d(X)
    assert len(ht_list) == 3
    ht1 = ht_list[0]
    ht2 = ht_list[1]
    ht3 = ht_list[2]

    X = X * 2
    if ht1 == 0:  # rosenbrock
        f = myrosenbrock(X)
    elif ht1 == 1:  # six hump
        f = mysixhumpcamp(X)
    elif ht1 == 2:  # beale
        f = mybeale(X)

    if ht2 == 0:  # rosenbrock
        f = f + myrosenbrock(X)
    elif ht2 == 1:  # six hump
        f = f + mysixhumpcamp(X)
    else:
        f = f + mybeale(X)

    if ht3 == 0:  # rosenbrock
        f = f + 5 * mysixhumpcamp(X)
    elif ht3 == 1:  # six hump
        f = f + 2 * myrosenbrock(X)
    else:
        f = f + ht3 * mybeale(X)

    y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])

    return y.astype(float)


def ackley5C(ht_list, X):
    z = np.concatenate([X, np.array(ht_list) * 0.125 - 1])
    a, b, c = 20, 0.2, 2 * np.pi
    f = -a * np.exp(-b * np.mean(z ** 2) ** 0.5) - np.exp(np.mean(np.cos(c * z))) + a + np.exp(1)

    y = f + 1e-6 * np.random.rand()

    return y.astype(float)


def svm_boston(ht_list, X):
    kernel_dict = {0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid'}
    gamma_dict = {0: 'scale', 1: 'auto'}
    shrinking_dict = {0: True, 1: False}
    kernel = kernel_dict[ht_list[0]]
    gamma = gamma_dict[ht_list[1]]
    shrinking = shrinking_dict[ht_list[2]]
    C = max(1e-4, min(10 ** X[0], 10))
    tol = max(1e-6, min(10 ** X[1], 1))
    nu = max(1e-6, min(10 ** X[2], 1))

    X, y = load_boston(return_X_y=True)

    n_cv = 5

    cv_rmse = np.zeros((n_cv, ))
    n_data = X.shape[0]
    n_train = int(n_data * 0.7)
    for cv in range(n_cv):
        inds = np.arange(n_data)
        np.random.RandomState(cv ** 2).shuffle(inds)
        train_inds = inds[:n_train]
        test_inds = inds[n_train:]
        train_X, train_y = X[train_inds], y[train_inds]
        test_X, test_y = X[test_inds], y[test_inds]
        regr = make_pipeline(StandardScaler(), NuSVR(kernel=kernel, gamma=gamma, shrinking=shrinking,
                                                     C=C, tol=tol, nu=nu))
        regr.fit(train_X, train_y)
        pred_test = regr.predict(test_X)
        cv_rmse[cv] = np.mean((pred_test - test_y) ** 2) ** 0.5

    return np.mean(cv_rmse).astype(float)


def xgboost_fashion_mnist(ht_list, X):
    booster_dict = {0: 'gbtree', 1: 'dart'}
    grow_policy_dict = {0: 'depthwise', 1: 'lossguide'}
    objective_dict = {0: 'multi:softmax', 1: 'multi:softprob'}
    max_depth = int(ht_list[0]) + 1
    booster = booster_dict[ht_list[1]]
    grow_policy = grow_policy_dict[ht_list[2]]
    objective = objective_dict[ht_list[3]]
    eta = max(1e-6, min(10 ** X[0], 1))
    gamma = max(1e-4, min(10 ** X[1], 10))
    subsample = max(1e-3, min(10 ** X[2], 1))
    reg_lambda = max(0, min(X[3], 5))

    params = {'max_depth': max_depth,
              'booster': booster,
              'grow_policy': grow_policy,
              'objective': objective,
              'eta': eta,
              'gamma': gamma,
              'subsample': subsample,
              'lambda': reg_lambda,
              'eval_metric': 'merror'}
    params['num_class'] = 10

    fashion_mnist = fetch_openml(data_id=40996, data_home=os.path.join(data_dir_root(), 'fashion_MNIST'), cache=True)
    input_data = fashion_mnist['data']
    output_data = fashion_mnist['target']

    train_inds = np.zeros((0, )).astype(int)
    test_inds = np.zeros((0, )).astype(int)
    for i in range(10):
        ilabel_data_inds = np.nonzero(output_data == str(i))[0]
        n_ilabel_train = int(0.7 * ilabel_data_inds.size)
        train_inds = np.concatenate([train_inds, ilabel_data_inds[:n_ilabel_train]])
        test_inds = np.concatenate([test_inds, ilabel_data_inds[n_ilabel_train:]])
    train_inds = sorted(train_inds)
    test_inds = sorted(test_inds)

    dtrain = xgb.DMatrix(data=input_data[train_inds], label=output_data[train_inds])
    dtest = xgb.DMatrix(data=input_data[test_inds], label=output_data[test_inds])

    results = {}
    xgb.train(params=params, dtrain=dtrain, num_boost_round=50,
              evals=[(dtest, 'eval'), (dtrain, 'train')], evals_result=results)
    print(results['eval']['merror'])
    return min(results['eval']['merror'])


class Func2CEvaluator(BlackboxEvaluator):
    def __init__(self, init_data_seed: Optional[int] = None, eval_init: bool = False):
        super().__init__(list_of_continuous=[(-1, 1), (-1, 1)], list_of_ordinal=[], list_of_nominal=[3, 5],
                         n_init_data=N_INIT_BENCHMARK, init_data_seed=init_data_seed, eval_init=eval_init)
        self.minimize = True
        self.info_str = 'Func2C'

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        result = x.new_empty(x.size(0))
        for i in range(x.size(0)):
            result[i] = float(func2C(ht_list=list(x[i, self.shift_nominal:].long().numpy()),
                                     X=x[i, :self.shift_ordinal].numpy()))
        return result.view(-1, 1)


class Func3CEvaluator(BlackboxEvaluator):
    def __init__(self, init_data_seed: Optional[int] = None, eval_init: bool = False):
        super().__init__(list_of_continuous=[(-1, 1), (-1, 1)], list_of_ordinal=[], list_of_nominal=[3, 5, 4],
                         n_init_data=N_INIT_BENCHMARK, init_data_seed=init_data_seed, eval_init=eval_init)
        self.minimize = True
        self.info_str = 'Func3C'

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        result = x.new_empty(x.size(0))
        for i in range(x.size(0)):
            result[i] = float(func3C(ht_list=list(x[i, self.shift_nominal:].long().numpy()),
                                     X=x[i, :self.shift_ordinal].numpy()))
        return result.view(-1, 1)


class Ackley5CEvaluator(BlackboxEvaluator):
    def __init__(self, init_data_seed: Optional[int] = None, eval_init: bool = False):
        super().__init__(list_of_continuous=[(-1, 1)], list_of_ordinal=[], list_of_nominal=[17, 17, 17, 17, 17],
                         n_init_data=N_INIT_BENCHMARK, init_data_seed=init_data_seed, eval_init=eval_init)
        self.minimize = True
        self.info_str = 'Ackley5C'

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        result = x.new_empty(x.size(0))
        for i in range(x.size(0)):
            result[i] = float(ackley5C(ht_list=list(x[i, self.shift_nominal:].long().numpy()),
                                       X=x[i, :self.shift_nominal].numpy()))
        return result.view(-1, 1)


class SVMBostonEvaluator(BlackboxEvaluator):
    def __init__(self, init_data_seed: Optional[int] = None, eval_init: bool = False):
        super().__init__(list_of_continuous=[(-4, 1), (-6, 0), (-6, 0)], list_of_ordinal=[], list_of_nominal=[4, 2, 2],
                         n_init_data=N_INIT_BENCHMARK, init_data_seed=init_data_seed, eval_init=eval_init)
        self.minimize = True
        self.info_str = 'SVMBoston'

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        result = x.new_empty(x.size(0))
        for i in range(x.size(0)):
            result[i] = float(svm_boston(ht_list=list(x[i, self.shift_nominal:].long().numpy()),
                                         X=x[i, :self.shift_nominal].numpy()))
        return result.view(-1, 1)


class XGBFashionMNISTEvaluator(BlackboxEvaluator):
    def __init__(self, init_data_seed: Optional[int] = None, eval_init: bool = False):
        super().__init__(list_of_continuous=[(-6, 0), (-4, 1), (-3, 0), (0, 5)],
                         list_of_ordinal=[10], list_of_nominal=[2, 2, 2],
                         n_init_data=N_INIT_BENCHMARK, init_data_seed=init_data_seed, eval_init=eval_init)
        self.minimize = True
        self.info_str = 'XGBFashionMNIST'

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        result = x.new_empty(x.size(0))
        for i in range(x.size(0)):
            result[i] = float(xgboost_fashion_mnist(ht_list=list(x[i, self.shift_ordinal:].long().numpy()),
                                                    X=x[i, :self.shift_ordinal].numpy()))
        return result.view(-1, 1)


if __name__ == "__main__":
    evaluator = XGBFashionMNISTEvaluator(0)
    value_list = []
    x_ = evaluator.generate_initial_data(n=10)[0]
    y_ = evaluator(x_)
    print(x_)
    print(y_)
    print(torch.min(y_).item(), torch.max(y_).item())

