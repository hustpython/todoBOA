import numpy as np 

bounds = {'l1':(1,10),
          'l2':(2,11),
          'l3':(0,1)}
dim = len(list(bounds.keys()))
keys = list(bounds.keys())
vbounds = np.array(list(bounds.values()), dtype=np.float)
random_state = np.random.RandomState()
data = np.empty((4,dim))
for col,(lower,upper) in enumerate(vbounds):
    data.T[col] = random_state.uniform(lower,upper,size=4)

params = dict(zip(keys,data[0]))
 
print(params)