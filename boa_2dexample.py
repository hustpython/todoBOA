#encoding:utf-8 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt import BayesianOptimization
#=====================定义Rastrigin函数===============
def Rastrigin(gridpoints):
    result = 0.0
    for i in range(0, gridpoints.shape[1]):
        result += np.power(gridpoints[:, i][:, None], 2) - \
            5*np.cos(2*np.pi*gridpoints[:, i][:, None])+5
    return result 


#===================定义二维数组的网格数据=============
length = 100
x1_grid = x2_grid = np.linspace(-3.2, 3.2, length)
x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
z_points = Rastrigin(grid)
z_points = z_points.reshape(length, length)
def plot3d():
    fig = plt.figure()
    axis = fig.gca(projection='3d')
    surf = axis.plot_surface(x1_grid, x2_grid,
                                z_points, rstride=1, cstride=1,
                                cmap=cm.jet, linewidth=0, antialiased=False,
                                alpha=1)

    plt.show()
#plot3d()
def target_Rastrigin(x1,x2,x3,x4):
    result = 0.0
    for i in [x1,x2,x3,x4]:
        result += np.power(i, 2) - \
            5*np.cos(2*np.pi*i)+5
    #return -result
    return 1.0/(1.0+result)
bo = BayesianOptimization(target_Rastrigin, {'x1': (-3,3),'x2':(-3,3),'x3':(-3,3),'x4':(-3,3)})
gp_params = {"alpha": 1e-3, "n_restarts_optimizer":3}
#bo.maximize(init_points=100,n_iter=300,acq='ucb',kappa=5)
#bo.maximize(init_points=50,n_iter=100,acq='ucb',kappa=20)
#bo.maximize(init_points=500,n_iter=200,acq="ei", xi=0.1, **gp_params)
#bo.maximize(init_points=50, n_iter=20, acq="poi", xi=0.1, **gp_params)
bo.maximize(init_points=5000, n_iter=50, acq='ucb',kappa=1)