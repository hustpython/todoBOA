#coding:utf-8 
import gplib 
import bolib
import numpy as np 

of = bolib.ofs.Branin()
data = {
    'X': np.zeros((2, len(of.get_bounds()))),
    'Y': np.array([[-1.0], [1.0]])
}

model = gplib.GP(
    gplib.mea.Constant(data),
    gplib.cov.SquaredExponential(data, is_ard=True),
    gplib.lik.Gaussian(is_noisy=True),
    gplib.inf.ExactGaussian()
)

fitting_method = gplib.fit.HparamOptimization(
    maxiter=100, restarts=1, ls_method="Powell")

af = bolib.afs.ExpectedImprovement(model)

seed = 48948
bo = bolib.methods.BayesianOptimization(model, fitting_method, af, seed)

x0 = bolib.util.random_sample(of.get_bounds(), batch_size=10)

bo.minimize(
    of.evaluate, x0,
    bounds=of.get_bounds(),
    tol=1e-7,
    maxiter=of.get_max_eval(),
    disp=True
)
