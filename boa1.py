import numpy as np 
from scipy.stats import norm 
from scipy.optimize import minimize
class oba(object):
    def __init__(self,target_func,vbounds):
        '''
           target:return the fitness of HFSS simulation.
           vabounds:return the variables and the consponsed bounds from the min to max.
        '''
        self.target_func = target_func 
        self.vbounds = vbounds 
        self.keys = list(vbounds.keys())
        self.vbounds_array = np.array(list(vbounds.values()), dtype=np.float)
        self.dim = len(self.keys)

        self.init_points = []
         
    def random_points(self,num,random=False):
        '''
           According the input vbounds,first initialize them in to big meshgrid.
           Example:
                  vbounds:{
                           L1:(2,10)
                           L2:(4:16)
                           L3:(3:9)
                           }
                  if every variable is divide to 3 meshgrid,then the multiplev-
                  ariables meshgrid has the number of 27:
                                   
        
        Example:
          
            >>> target_func = lambda p1, p2: p1 + p2
            >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
            >>> space = TargetSpace(target_func, pbounds, random_state=1)
            >>> space.random_points(3)
            array([[ 55.33253689,   0.54488318],
                [ 71.80374727,   0.4236548 ],
                [ 60.67357423,   0.64589411]])
        '''
        data = np.empty((num,self.dim))
        if random:
           random_state = np.random.RandomState()
           for col,(lower,upper) in enumerate(self.vbounds_array):
               data.T[col] = random_state.uniform(lower,upper,size=num)
        else:
            pass 
        return data 
    def init(self,init_points):

        rand_points = self.random_points(3)
        self.init_points.extend(rand_points)
        for x in self.init_points:
            y = self._observe_point(x)


    def _observe_point(self,x):
        x = np.asarray(x).ravel()
        assert x.size == self.dim 
        
        params = dict(zip(self.keys,x))

        y = self.target_func(**params)

        return y

    def maximize(self,
                 init_point=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        pass   

#An object to compute the acquisition functions.
    """
    If UCB is to be used,a constant kappa is needed.
    """
    @staticmethod
    def _ucb(x,gp,kappa):
        mean,std = gp.predict(x,return_std=True)
        return mean + kappa * std 
    @staticmethod
    def _ei(x,gp,y_max,xi):
        mean,std = gp.predict(x,return_std=True)
        z = (mean - y_max-xi)/std 
        return (mean-y_max-xi)*norm

    @staticmethod
    def _poi(x,gp,y_max,xi):
        mean,std = gp.predict(x,return=True)
        z = (mean - y_max - xi)/std 
        return norm.cdf(z)

    def acq_max(ac,gp,y_max,bounds,random_state,
                n_warmup=1000000,n_iter=250)
        '''
        A function to find the maximum of the acquisition function

        It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
        optimization method. First by sampling `n_warmup`(1e5) points at random,
        and then running L-BFGS-B from `n_iter`(250) random starting points.
        
        Parameters...
        '''
        x_tries = random_state.uniform(bounds[:,0],bounds[:,1],
                                       size=(n_iter,bounds.shape[0]))

        ys = ac(x_tries)