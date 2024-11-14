import numpy                as np
import os, sys
import pandas as pd
from scipy.stats import truncnorm, norm
from scipy.optimize import minimize

#The computer code takes q variables $(\lambda_1, \lambda_2, \dots, \lambda_q)$. $p$ of them are physical variables to calibrate and are teated with uniform prior, and $q-p$ are numerical variables to represent the model error and are treated with hierarchical model.
#The array "index_lambda_p" indicates which of these variables are the physical variables and "index_lambda_q". 

# transform_Lambda takes a normalized vector of parameters $\lambda_{norm}$, with the first $p$ variables treated as physical variable, and the next $q-p$ as the numerical variables, and return the vector $\boldsymbol{\lambda}$, with physical values and with each variable at the right position relatively to "index_lambda_p" and "index_lambda_q". 

# p_lambda_df takes a dataframe df_Lambda of vectors $(\lambda_k)_{k=1}^M$ and a vector of hyperparameters $\alpha$, and returns the vector $p_{\\Lambda}(\lambda_k \mid \alpha)_{k=1}^M$

# get_likelihoods_dflambda takes a dataframe df_Lambda of vectors $(\lambda_k)_{k=1}^M$, and return the simulations, the standard deviations associated if gaussian processes are used, and the likelihoods associated to each $\boldsymbol{\lambda}_k.$ The argument sigma refers to the observations standard deviation; mm_list gathers the parameters of the GPs, results_measures is the dataframe of the observations; index indicates the output considered for the computation of the likelihood (if multiple outputs are indicated, one likelihood will be computed for each output and each $\lambda_k$); std_code indicates whether or not the standard deviation of the code should be considered, and idx_loo is the index of the observation that should be removed for the likelihood computation in the LOO scheme.

# sample_Lambda takes an integer $M$ and a vector of hyperparemters $\alpha$ and returns a sample $(\lambda_k)_{k=1}^M$ i.i.d. with density $p_{\Lambda}(.\mid \alpha)$.

# likelihood_alpha takes a vector $\alpha$, the likelihoods computed with another vector $\alpha^\star$, $p_{\Lamba}(\lambda_k\mid \alpha^\star)_{k=1}^{n}$ the prior densities computed with $\alpha^\star$, and the dataframe $(\lambda_k)_{k=1}^M$ sampled with $p_{\Lambda}(.\mid \alpha^\star)$. It returns the estimated likelihood of $\alpha$ with importance sampling.

# find_best takes $\alpha^\star$, the likelihoods computed with $\alpha^\star$, the sample $(\lambda_k)_{k=1}^M$ i.i.d with density $p_{\Lambda}(.\mid \alpha^\star)$, and returns the estimated maximum a posteriori, considering uniform prior.

def dist2(y1,y2):
    return np.sqrt(np.mean((y1-y2)**2/y2**2))

def transform_Lambda(Lambda, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda):
    Lambda_new = [Lambda[(index_lambda_p + index_lambda_q).index(x)] for x in range(len(index_lambda_p + index_lambda_q))] #reorder vector
    Lambda_new = np.array([Lambda_new[x]*(bMAXlambda[x] - bMINlambda[x])+bMINlambda[x] for x in range(len(Lambda_new))]) #back to physical values
    return(Lambda_new)

def p_lambda_df(df_Lambda, alpha, index_lambda_p, index_lambda_q, scale, bMINlambda, bMAXlambda):
    scale = np.array([scale]*len(alpha)) #same scale for each variable
    lambd_norm = (df_Lambda-bMINlambda)/(bMAXlambda - bMINlambda) #normalize values
    def fun1(x): return all(0 <= coord <= 1 for coord in x) 
    coeff1 = lambd_norm.iloc[:,index_lambda_p].apply(fun1, axis=1) #Uniform [0,1] for the coordinates "index_lambda_p"
    a, b = (0 - alpha) / scale, (1 - alpha) / scale 
    coeff2=1
    for ii in range(len(alpha)):
        coeff2 = coeff2*truncnorm.pdf((lambd_norm.iloc[:,index_lambda_q[ii]].values - alpha[ii])/scale[ii], a[ii],b[ii])/scale[ii] #truncated gaussian for the coordinates "index_lambda_q"
    return coeff1*coeff2

def get_likelihoods_dflambda(df_Lambda, sigma,results_measures, myCODE, mm_list = None, index = [1], std_code = False, idx_loo = None):
    Ysimu = myCODE(df_Lambda, index = index,  std_bool = std_code, vectorize = True, idx_loo = idx_loo,  mm_list = mm_list) #Get simulations
    if std_code: #if gaussian process regression
        Ysimu, Ystd = Ysimu #Get std deviations and simulations
        res = [[np.prod(norm.pdf(results_measures[list(set(range(len(results_measures))) - set([idx_loo]))]-Ysimu[iii].iloc[:,0], loc=0, scale=np.sqrt(sigma + Ystd[iii].iloc[:,0])))] for iii in range(len(Ysimu))] #compute gaussian likelihoods, considering std of observation noise and std of gaussian process
        
    else: #if deterministic simulator
        res = [[np.prod(norm.pdf(results_measures[list(set(range(len(results_measures))) - set([idx_loo]))]-Ysimu[iii].iloc[:,0], loc=0, scale=sigma))] for iii in range(len(Ysimu))] #compute gaussian likelihoods, considering only observation noise
        Ystd = None
    return Ysimu, Ystd, np.array(res)

def sample_Lambda(alpha, M, index_lambda_p, index_lambda_q, scale, bMINlambda, bMAXlambda):
    Lambda_list = []
    if len(index_lambda_q) > 0:
        scale = np.array([scale]*len(alpha))
        a, b = (0 - alpha) / scale, (1 - alpha) / scale
    for k in range(M):
        if len(index_lambda_q) > 0:
            sample_lambda_q = np.array([truncnorm.rvs(a[ii], b[ii], size=1)[0]*scale[ii] + alpha[ii] for ii in range(len(alpha))]) #truncated gaussian sample
            Lambda = transform_Lambda(np.concatenate([np.random.uniform(0,1,len(index_lambda_p)), sample_lambda_q]), index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) #concatenate uniform sample with truncated gaussian sample, and use transform_Lambda to reorder and get physical values
        else: 
            Lambda = transform_Lambda(np.random.uniform(0,1,len(index_lambda_p)), index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) #only uniform sample
        Lambda_list.append(Lambda)
    return pd.DataFrame(np.array(Lambda_list))

    
def likelihood_alpha(alpha, likelihoods_alpha_star, denom_is, df_Lambda, index_lambda_p, index_lambda_q, scale,bMINlambda, bMAXlambda):
    ratio_is = np.array(p_lambda_df(df_Lambda = df_Lambda, alpha = alpha, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, scale = scale,bMINlambda = bMINlambda, bMAXlambda = bMAXlambda)/denom_is) #compute importance sampling ratios
    ratio_is = ratio_is.reshape(len(ratio_is),1)
    return np.mean(likelihoods_alpha_star*ratio_is) #Mean of the likelihoods weighted by the importance sampling ratios


def find_map(alpha_star, bounds, likelihoods_alpha_star, df_Lambda, index_lambda_p, index_lambda_q, scale, bMINlambda, bMAXlambda):
    denom_is = p_lambda_df(df_Lambda = df_Lambda, alpha = alpha_star, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, scale = scale,bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) #Compute the denomination of importance sampling ratio
    fun = lambda alpha: likelihood_alpha(alpha = alpha, likelihoods_alpha_star = likelihoods_alpha_star, denom_is = denom_is, df_Lambda = df_Lambda, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, scale = scale,bMINlambda = bMINlambda, bMAXlambda = bMAXlambda)
    baseline = fun(alpha_star)
    fun_opt = lambda alpha: -fun(alpha)/baseline #Normalized by baseline so that values of functions are not too low
    return minimize(fun_opt, alpha_star, method='L-BFGS-B', bounds=bounds).x #minimize the opposite with L-BFGS-B

def save_results(data, file, pre_path, calib = None): #function to save the results obtained with each strategy
    if(os.path.isdir(pre_path)==False):  
        os.mkdir(pre_path)
    if(os.path.isdir(pre_path + f"/calib_{calib}")==False): 
        os.mkdir(pre_path + f"/calib_{calib}")
    data.to_csv(pre_path +f"/calib_{calib}/{file}")
