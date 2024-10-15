import numpy as np
import pandas as pd
from utils_calib import *

#The function func_alphamap is the implementation of the algorithm to estimate alpha_map

def norm2(y1,y2):
    return np.sqrt(np.sum((y1-y2)**2))

def check_repeat(stored_alpha, new_alpha, threshold): #check if an alpha of the df stored_alpha is very close to new_alpha
    if len(stored_alpha)==0: return False
    else: return np.min(np.apply_along_axis(lambda xx: norm2(xx, new_alpha),1, stored_alpha)) < threshold


def func_alphamap(index_calib, M, iter_lim, threshold, alpha_min, alpha_max, delta_alpha, scale, results_measures, sigma, myCODE, mm_list, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda, pre_path, loo = True, std_code = True):
    if not loo: list_idx_loo = [None]
    else: list_idx_loo = range(len(results_measures))
    alpha_df = np.zeros((len(list_idx_loo), len(index_lambda_q))) #the alpha_map will be stored here
    sample_size_df = []
    for idx_loo in list_idx_loo: #for each observation x_j
        print(f"IDX LOO  {idx_loo}")

        alpha_new = np.array([0.5]*len(index_lambda_q)) #initial alpha_star
        alpha_star = np.array([10**6]*len(index_lambda_q))

        iter=1
    
        stored_alpha = np.empty((0,len(alpha_star))) #all the alpha_star computed will be stored here
        M_used = M #number of i.i.d. realizations

        while iter <= iter_lim and norm2(alpha_new,alpha_star)> threshold: #while not fixed point
            alpha_star = alpha_new.copy()
            np.random.seed(123456)
            bounds = [(max(alpha_star[ii] - delta_alpha,alpha_min), min(alpha_max, alpha_star[ii]+delta_alpha)) for ii in range(len(alpha_star))] #bounds for the optimization
            df_Lambda = sample_Lambda(alpha = alpha_star, M = M_used, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q,scale=scale,bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) #sample lambda
            Ysimu_list, Ystd_list, stored_likelihoods = get_likelihoods_dflambda(df_Lambda = df_Lambda.values, sigma = sigma, myCODE = myCODE, mm_list = mm_list, results_measures = results_measures, index=[index_calib], std_code = std_code, idx_loo = idx_loo) #compute likelihood of each lambda
            alpha_new = find_map(alpha_star = alpha_star, bounds = bounds, likelihoods_alpha_star = stored_likelihoods, df_Lambda = df_Lambda, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, scale = scale,bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) #optimize a posterior distribution
            iter = iter + 1
            if check_repeat(stored_alpha, alpha_new, threshold): #if this alpha already encountered, increase size of sampling to prevent infinite loop
                M_used = M_used + 2000
            stored_alpha = np.concatenate([stored_alpha, alpha_new.reshape(1,len(alpha_new))], axis = 0) 
        alpha_star = alpha_new.copy()

        if idx_loo is None: alpha_df[0] = alpha_star
        else: alpha_df[idx_loo] = alpha_star
        sample_size_df.append(M_used-2000)
        
    save_results(pd.DataFrame(alpha_df), "alpha_df.csv", pre_path = pre_path, calib = index_calib)
    save_results(pd.DataFrame(sample_size_df), "sample_sizes.csv", pre_path = pre_path, calib = index_calib)
    