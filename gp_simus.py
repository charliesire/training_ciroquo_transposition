import numpy as np
import pandas as pd


#The parameters of the gaussian processes were optimized with pylibkriging package, and are stored in a list mm_list. We have one GP in $\lambda$ for each $x_j$ and each output $y_t$. Then mm_list has 10 elements (one for each $x_j$), and each one is a list of 3 elements (one for each $y_t$). 
#And each mm_list[j][t] is a dictionnary mm with:
#- mm["invK"] the inverse of the covariance matrix
#- mm["beta"] the trend 
#- mm["X"] the training input database $(\lambda_k)$
#- mm["y"] the training output database
#- mm["theta"] the ranges
#- mm["sigma2"] the variance

#The function matern_5_2 is the computation of the matern 5/2 kernel between a point "x1" and a dataframe "df", with ranges "theta" and variance "sigma2".

#The function pred_np takes as argument an array of vectors "lambd", "idx_xinput" the index $j$ associated to the observation point $x_j$ to consider, "idx_output" the index $t$ associated to the output $y_t$ to consider$, "std_bool" a boolean indicating whether or not the standard deviation prediction is considered, and "vectorize" a boolean indicating whether "lambd" is a unique vector of input parameters or an array or multiple inputs. It returns the gaussian process prediction, with the standard deviation is std_bool = True.

#The function myCODE takes as argument an array of vectors "lambd", "index" the indices of the outputs that we want to get ($y_1$ and/or $y_2$ and/or $y_3$), "std_bool" a boolean indicating whether or not the standard deviation prediction is considered, "vectorize" a boolean indicating whether "lambd" is a unique vector of input parameters or an array or multiple inputs, "idx_loo" is the index $j$ associated with the observation point that should be removed if new_x is False. If new_x is True, the outputs are computed only at idx_loo. If new_x is False, it returns a list of $M$ dataframes of $n \times c$ predictions, where $n=10$ if idx_loo = None and $n=9$ otherwise, $c$ is the length of "index", and $M$ is the number of vectors in "lambd". If new_x is True, it returns a dataframe $M \times c$.

def matern_5_2(x1, df, theta=1, sigma2=1):
    diff_theta = np.sqrt(np.sum((x1 - df)**2 / theta**2, axis = 1))
    res = (1.0 + np.sqrt(5.0) * diff_theta + (5.0 / 3.0) * diff_theta**2)*np.exp(-np.sqrt(5.0) * diff_theta)
    return sigma2*res
    
def pred_np(mm_list, lambd, idx_xinput, idx_output, std_bool = False, vectorize = True):
    if not vectorize:
        lambd = np.array([lambd])
    mm = mm_list[idx_xinput][idx_output] #Take the dictionnary associated with idx_xinput and idx_output
    K_inverse = mm["invK"] #Inverse covariance matrix
    mu = mm["beta"][0][0] # Trend
    points = mm["X"] #Training input database
    Ymod = mm["y"] #Training output database
    theta = np.array([mm["theta"][i][0] for i in range(len(points[0]))]) #Ranges
    sigma2 = mm["sigma2"] #variance
    covariances = np.transpose(np.concatenate([[matern_5_2(x1 = lambd[i], df = points, theta = theta, sigma2 = sigma2)] for i in range(len(lambd))])) #covariance vector
    prod_mat_1 = np.dot(np.transpose(covariances), K_inverse) 
    prod_mat_2 = mu + np.dot(prod_mat_1, Ymod-mu) #prediction mean
    if not std_bool:
        return prod_mat_2.flatten()
    else:
        prod_mat_3 = np.concatenate([[np.dot(prod_mat_1[i,:],covariances[:,i])] for i in range(len(prod_mat_1))]) 
        return prod_mat_2.flatten(), np.sqrt(sigma2-prod_mat_3) # prediction mean and prediction standard deviation



def myCODE(lambd, index = [1,2,3],  std_bool = False, vectorize = True, idx_loo = None, new_x = False, mm_list = None):
    len_final = len(list(set(range(10)) - set([idx_loo]))) #Number of observations
    if not vectorize:
        res = np.empty((len_final,0))
        res_std = np.empty((len_final,0))
        for y in index: #for each output
            Ysimu = [pred_np(mm_list,lambd, xx, int(y-1), std_bool = std_bool, vectorize=vectorize) for xx in list(set(range(10)) - set([idx_loo]))] #get the prediction for all x_j
            if std_bool: #if std deviation of the prediction to consider
                Ysimu, Ystd = np.array([Ysimu[k][0] for k in range(len(Ysimu))]).reshape(-1,1), np.array([Ysimu[k][1] for k in range(len(Ysimu))]).reshape(-1,1) 
                res_std = np.concatenate([res_std, Ystd], axis=1) #concatenate the obtained columns (1 per output)
            res = np.concatenate([res, np.array(Ysimu)],axis=1) #concatenate the obtained columns (1 per output)
        res = pd.DataFrame(res)
        res_std = pd.DataFrame(res_std)
    else:
        if not new_x:
            res = [np.empty((len_final,0)) for _ in range(len(lambd))]
            res_std = [np.empty((len_final,0)) for _ in range(len(lambd))]
            for y in index: #for each output
                Ysimu = [pred_np(mm_list, lambd, xx, int(y-1), std_bool = std_bool, vectorize=vectorize) for xx in list(set(range(10)) - set([idx_loo]))] #get the prediction for all x_j
                if std_bool: #if std deviation of the prediction to consider
                    Ysimu, Ystd = np.array([Ysimu[k][0] for k in range(len(Ysimu))]), np.array([Ysimu[k][1] for k in range(len(Ysimu))])  
                    res_std = [np.concatenate([res_std[ii], Ystd[:,ii].reshape(-1,1)], axis=1) for ii in range(len(lambd))] #concatenate the obtained columns for each lambda (1 per output) 
                res = [np.concatenate([res[ii], np.array(Ysimu)[:,ii].reshape(-1,1)], axis=1) for ii in range(len(lambd))] #concatenate the obtained columns for each lambda (1 per output) 
            res = [pd.DataFrame(rr) for rr in res] #for each lambda get dataframe
            res_std = [pd.DataFrame(rr) for rr in res_std] #for each lambda get dataframe
        else: #if new_x
            res = np.empty((len(lambd),0))
            res_std = np.empty((len(lambd),0))
            for y in index: #for each output
                Ysimu = pred_np(mm_list, lambd = lambd, idx_xinput = idx_loo, idx_output = int(y-1), std_bool = std_bool, vectorize=True) #get the predictions at idx_loo
                if std_bool: 
                    Ysimu, Ystd = Ysimu
                    res_std = pd.DataFrame(np.concatenate([res_std, Ystd.reshape(-1,1)], axis = 1)) #concatenate the obtained columns
                res = pd.DataFrame(np.concatenate([res, Ysimu.reshape(-1,1)], axis=1)) #concatenate the obtained columns
    if std_bool:
        return res, res_std
    else: return res

