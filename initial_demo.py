#coding:utf-8 
import numpy as np 
from bayes_opt import BayesianOptimization
def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
bo = BayesianOptimization(target, {'x': (-2, 10)})
bo.maximize(init_points=1, n_iter=100, acq='ucb', kappa=5)
