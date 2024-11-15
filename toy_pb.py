import numpy as np
import pandas as pd

# We work with the functions f_1(x, \lambda) = (0.2+\lambda_1)*ln(1+x) + sqrt(\lambda_2) and f_2(x, \lambda) = sin(1/2*(1+\lambda_1))exp(-1/(1+x)) + \lambda_2^0.15.
# Here the observations points are x_j = j/6 for j=1,..,5. Then we work with n=5 observations
#The function myCODE takes as argument an array of vectors "lambd", "index" the indices of the outputs that we want to get ($y_1$ and/or $y_2$), "std_bool" a boolean indicating whether or not the standard deviation prediction is considered, "vectorize" a boolean indicating whether "lambd" is a unique vector of input parameters or an array or multiple inputs, "idx_loo" is the index $j$ associated with the observation point that should be removed if new_x is False. If new_x is True, the outputs are computed only at idx_loo. If new_x is False, it returns a list of $M$ dataframes of $n \times c$ predictions, or $(n-1) \times c$ if idx_loo is not None, where $c$ is the length of "index", and $M$ is the number of vectors in "lambd". If new_x is True, it returns a dataframe $M \times c$. 

def func(x, lambd, idx_y, vectorize):
    if not vectorize: 
        if idx_y == 1: return np.array([(0.2+lambd[0])*np.log(1+x)+ lambd[1]**0.5])
        else: return np.array([np.sin(0.5+lambd[0]/2)*np.exp(-1/(1+x))+ lambd[1]**0.15])
    else:
        if idx_y == 1: return (0.2+lambd[:,0])*np.log(1+x)+ lambd[:,1]**0.5
        else: return np.sin(0.5+lambd[:,0]/2)*np.exp(-1/(1+x)) + lambd[:,1]**0.15


def myCODE(lambd, index = [1,2], std_bool = False, vectorize = True, idx_loo = None, new_x = False, mm_list = None):
    x_vec = np.linspace(0,1,7)[1:-1]
    len_final = len(list(set(range(5)) - set([idx_loo]))) #Number of observations
    if not vectorize:
        res = np.empty((len_final,0))
        res_std = np.empty((len_final,0))
        for y in index: #for each output
            Ysimu = [func(x_vec[xx], lambd, y,vectorize)  for xx in range(len(x_vec)) if xx != idx_loo] #get the prediction for all x_j
            res = np.concatenate([res, np.array(Ysimu)],axis=1) #concatenate the obtained columns (1 per output)
        res = pd.DataFrame(res)
        res_std = pd.DataFrame(res_std)
    else:
        if not new_x:
            res = [np.empty((len_final,0)) for _ in range(len(lambd))]
            res_std = [np.empty((len_final,0)) for _ in range(len(lambd))]
            for y in index: #for each output
                Ysimu = [func(x_vec[xx], lambd, y,vectorize) for xx in range(len(x_vec)) if xx != idx_loo] #get the prediction for all x_j
                res = [np.concatenate([res[ii], np.array(Ysimu)[:,ii].reshape(-1,1)], axis=1) for ii in range(len(lambd))] #concatenate the obtained columns for each lambda (1 per output) 
            res = [pd.DataFrame(rr) for rr in res] #for each lambda get dataframe
            res_std = [pd.DataFrame(rr) for rr in res_std] #for each lambda get dataframe
        else: #if new_x
            res = np.empty((len(lambd),0))
            res_std = np.empty((len(lambd),0))
            for y in index: #for each output
                Ysimu = func(x_vec[idx_loo], lambd, y,vectorize)
                res = pd.DataFrame(np.concatenate([res, Ysimu.reshape(-1,1)], axis=1)) #concatenate the obtained columns

    return res


def myCODEnew(x, lambd, index = 1, std_bool = False, mm_list = None):
    return func(x = x, lambd = lambd, idx_y = index,vectorize = True) 
    
