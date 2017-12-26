#encoding:utf-8 
import numpy as np 
from bayes_opt import BayesianOptimization 

def target_Rastrigin(x1,x2,x3):
    result = 0.0
    for i in [x1,x2,x3]:
        result += np.power(i, 2) - \
            5*np.cos(2*np.pi*i)+5
    return 1.0/(1.0+result) 
def target2(x):
    res = x + 10*np.sin(5*x) + 7*np.cos(4*x)
    return res 
print(target2(7.8568))
bo = BayesianOptimization(target_Rastrigin, {'x1': (-3.2, 3.2),'x2':(-3.2,3.2),'x3':(-3.2,3.2)})
#bo = BayesianOptimization(target2, {'x': (0,9)})
gp_params = {"alpha": 1e-3, "n_restarts_optimizer": 2}
#bo.maximize(init_points=50,n_iter=30,acq='ucb',kappa=5,**gp_params)
#bo.maximize(init_points=50,n_iter=100,acq='ucb',kappa=20)
#bo.maximize(init_points=50,n_iter=200,acq="ei", xi=0.1, **gp_params)
bo.maximize(init_points=50, n_iter=100, acq="poi", xi=0.1, **gp_params)
#bo.maximize(init_points=100, n_iter=50, kappa=5)