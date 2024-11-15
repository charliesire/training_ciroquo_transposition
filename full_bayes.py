import numpy as np
import pandas as pd
from scipy.stats import norm
from utils_calib import * 

#The function full_bayes_estimator takes a matrix YY that gathers the samples h(\lambda_k) for a given function h, ratio is that gathers the importance sampling ratios p(\lambda_k\mid alpha_i)/p(\lambda_k\mid alpha_map), and return the estimator EN,M(h(\Lambda). In other words, this function estimates the expectation a posterior E(h(\Lambda) \mid yobs).

#The function plot_transpo_fullbayes computes the prediction mean and standard deviation, from the "dic_Lambda" the list of samples of posterior samples $\boldsymbol{\Lambda}$ obtained for each idx_loo, "dic_alpha" the list of posterior samples of $alpha$ for each idx_loo, "Ysimu_list" and "Ystd_list" the GP means and standard deviations, and "alpha_df" the $\boldsymbol{\alpha}_{\text{MAP}}$ obtained for all idx_loo. 

# The function compute_error_fullbayes computes the RMSRE and the levels of the prediction intervals from ratio_is (same argument as full_bayes_estimator) and Ysimu_list and Ystd_list the GP means and standard deviations and the true_values

# The function full_bayes_results uses for each observation variable, plot_transpo_fullbayes and compute_error_fullbayes to compute the performance metrics
def full_bayes_estimator(YY, ratio_is,loo):
    if loo:
        res = []
        for ii in range(len(ratio_is)): #for each observation point $x_j$
            l1 = []
            for kk in range(ratio_is[ii].shape[1]): #for each alpha sample
                l1.append(np.average(YY.iloc[ii,:], weights = ratio_is[ii][:,kk])) #get the weighted mean, where the weights are the importance sampling ratios
            res.append(np.mean(l1)) #get the overall mean over 
        return np.array(res)
    else: 
        arr = np.array([np.apply_along_axis(lambda vec: np.average(vec, weights = ratio_is[:,kk]), 1, YY) for kk in range(ratio_is.shape[1])])
        return np.mean(arr, axis=0)
    

def plot_transpo_fullbayes(dic_Lambda, dic_alpha, Ysimu_list, Ystd_list, alpha_df, index_lambda_p, index_lambda_q, scale, bMINlambda, bMAXlambda,loo):
    if not loo: list_idx_loo = [None]
    else: list_idx_loo = range(len(dic_Lambda))
    
    predictions = pd.DataFrame(np.zeros((Ysimu_list[0].shape[0], Ysimu_list[0].shape[1])))
    predictions_square = pd.DataFrame(np.zeros((Ysimu_list[0].shape[0], Ysimu_list[0].shape[1])))
    denom_ratio_is = [p_lambda_df(df_Lambda = dic_Lambda[ii], alpha = alpha_df[ii], index_lambda_p=index_lambda_p, index_lambda_q = index_lambda_q, scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) for ii in range(len(dic_Lambda))] #compute the denominator of the importance sampling ratios
    ratio_is = [np.concatenate([p_lambda_df(df_Lambda = dic_Lambda[ii], alpha = alpha_prov, index_lambda_p=index_lambda_p, index_lambda_q = index_lambda_q, scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda).values.reshape(-1,1) for alpha_prov in dic_alpha[ii]], axis=1)/denom_ratio_is[ii].values.reshape(-1,1) for ii in range(len(dic_Lambda))] #compute the importance sampling ratios
    if not loo: ratio_is = ratio_is[0]
    for index_predict in range(1,Ysimu_list[0].shape[1]+1):
        YY = pd.concat([pd.DataFrame(Ysimu_list[ii].iloc[:, index_predict-1]) for ii in range(len(Ysimu_list))], axis=1) # get all the output values (dim M x n)
        pred = full_bayes_estimator(YY = YY, ratio_is = ratio_is, loo = loo) #compute the prediction mean
        pred_square = full_bayes_estimator(YY = YY**2, ratio_is = ratio_is, loo = loo) 
        if Ystd_list is not None: 
            std_mm = pd.concat([pd.DataFrame(Ystd_list[ii].iloc[:, index_predict-1]) for ii in range(len(Ystd_list))], axis=1) # get all the stds values (dim M x n)
            pred_square = pred_square + full_bayes_estimator(YY = std_mm**2, ratio_is = ratio_is, loo = loo)   # equation 19 line 2 part 1
        predictions.iloc[:, index_predict-1] = pred # result agregation
        predictions_square.iloc[:, index_predict-1] = pred_square # result agregation
    
    std_pred = np.sqrt(predictions_square - predictions**2)  #compute the prediction standard deviation
        
    return pd.concat([predictions, std_pred]), ratio_is

def compute_error_fullbayes(ratio_is, Ysimu_list, true_values, loo, Ystd_list = None):
    predictions = pd.DataFrame(np.zeros((Ysimu_list[0].shape[0], Ysimu_list[0].shape[1]))) 
    predictions_square = pd.DataFrame(np.zeros((Ysimu_list[0].shape[0], Ysimu_list[0].shape[1])))
    res = []
    res_intervals = np.empty((Ysimu_list[0].shape[0],0))
    for index_predict in range(1,Ysimu_list[0].shape[1]+1): #for each output
        YY = pd.concat([pd.DataFrame(Ysimu_list[ii].iloc[:, index_predict-1]) for ii in range(len(Ysimu_list))], axis=1) # get all the output values (dim M x n)
        if Ystd_list is not None: std_mm = pd.concat([pd.DataFrame(Ystd_list[ii].iloc[:, index_predict-1]) for ii in range(len(Ystd_list))], axis=1) # get all the stds values (dim M x n)
        pred = full_bayes_estimator(YY = YY, ratio_is = ratio_is, loo = loo) #compute the prediction mean
        eta = abs(pred-true_values[f"Y{index_predict}"]).values
        error = dist2(pred,true_values[f"Y{index_predict}"])
        res.append(error)
        if Ystd_list is not None: diff_cdf = pd.DataFrame(norm.cdf(((pred+eta).reshape(-1,1) - YY)/std_mm) - norm.cdf(((pred-eta).reshape(-1,1) - YY)/std_mm))    # In of Equation 22
        else: diff_cdf = pd.DataFrame(abs(pred.reshape(-1,1) - YY) < eta.reshape(-1,1))   # In of Equation 22
        intervals = full_bayes_estimator(YY = diff_cdf, ratio_is = ratio_is, loo = loo).reshape(-1,1) #compute the levels of the predictions intervals : full equation 22
        res_intervals = np.concatenate([res_intervals, intervals], axis=1)
    return res, res_intervals


def full_bayes_results(index_calib, index_lambda_p, index_lambda_q, scale, bMINlambda, bMAXlambda, pre_path, true_values,loo = True, std_code = True):
    if not loo: list_idx_loo = [None]
    else: list_idx_loo = range(len(true_values))
    alpha_df = pd.read_csv(pre_path + f"/calib_{index_calib}/alpha_df.csv", index_col=0).values #get alpha_map for each idx_loo
    if loo:
        dic_alpha = [pd.read_csv(pre_path + f"/calib_{index_calib}/samples_alpha_post_{ii}.csv", index_col=0).values for ii in list_idx_loo] #get the samples alpha for each idx_loo
        dic_Lambda = [pd.read_csv(pre_path + f"/calib_{index_calib}/lambd_post_{ii}.csv", index_col=0)for ii in list_idx_loo]#get the samples lambda for each idx_loo
    else: 
        dic_alpha = [pd.read_csv(pre_path + f"/calib_{index_calib}/samples_alpha_post.csv", index_col=0).values for ii in list_idx_loo] #get the samples alpha for each idx_loo
        dic_Lambda = [pd.read_csv(pre_path + f"/calib_{index_calib}/lambd_post.csv", index_col=0)for ii in list_idx_loo]#get the samples lambda for each idx_loo
    YY = pd.read_csv(pre_path + f"/calib_{index_calib}/Ysimu_list.csv", index_col = 0) #get the output values
    Ysimu_list = np.array_split(YY, len(YY) // len(true_values))# Crée une liste de DataFrames
    if std_code:
        Ystd = pd.read_csv(pre_path + f"/calib_{index_calib}/Ystd_list.csv", index_col = 0) #get the stds values
        Ystd_list = np.array_split(Ystd, len(Ystd) // len(true_values))
    else: Ystd_list = None
    plot1 = plot_transpo_fullbayes(dic_Lambda = dic_Lambda, dic_alpha = dic_alpha, Ysimu_list = Ysimu_list, Ystd_list = Ystd_list, alpha_df = alpha_df, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda, loo = loo) #compute prediction mean and std
    save_results(plot1[0], "plot_full_bayes.csv", pre_path = pre_path, calib = index_calib) 
    ratio_is = plot1[1]
    errors_bayes, intervals_bayes = compute_error_fullbayes(ratio_is = ratio_is, Ysimu_list = Ysimu_list, Ystd_list = Ystd_list, true_values = true_values, loo = loo) #compute RMSRE and levels of prediction intervals
    save_results(pd.DataFrame(errors_bayes), "errors_bayes.csv", pre_path = pre_path, calib = index_calib)
    save_results(pd.DataFrame(intervals_bayes), "conf_level_bayes.csv", pre_path = pre_path, calib = index_calib)

def predictor(x, index, sample_alpha, sample_lambda, myCODEnew, alpha_star = None,  scale, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda, std_bool = False, mm_list = None, mean_std = True, psi = None):
    if std_bool & (psi is not None): return "Error: Can not compute expectation of psi(Y) if std_bool is True"
    if psi is None and mean_std == False: return "Nothing to compute as mean_std is False and psi is None"
    if std_bool: Ysimu, Ystd = myCODEnew(x = x, lambd = sample_lambda, index = index, std_bool = std_bool, mm_list = mm_list)
    else: Ysimu = myCODEnew(x = x, lambd = sample_lambda, index = index, std_bool = std_bool, mm_list = mm_list)
    if alpha_star is None: 
      ratio_is = np.array([1]*len(Ysimu)).reshape(-1,1)
    else:
      denom_ratio_is = p_lambda_df(df_Lambda = pd.DataFrame(sample_lambda), alpha = alpha_star, index_lambda_p=index_lambda_p, index_lambda_q = index_lambda_q, scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda)  #compute the denominator of the importance sampling ratios
      ratio_is = np.concatenate([p_lambda_df(df_Lambda = pd.DataFrame(sample_lambda), alpha = alpha_prov, index_lambda_p=index_lambda_p, index_lambda_q = index_lambda_q, scale = scale, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda).values.reshape(-1,1) for alpha_prov in sample_alpha], axis=1)/denom_ratio_is.values.reshape(-1,1)  #compute the importance sampling ratios
    if mean_std:
        mean = np.mean(np.array([np.average(Ysimu, weights = ratio_is[:,kk]) for kk in range(ratio_is.shape[1])]))
        std = np.sqrt(np.mean(np.array([np.average(Ysimu**2, weights = ratio_is[:,kk]) for kk in range(ratio_is.shape[1])])) - mean**2)
        if std_bool: std = std + np.mean(np.array([np.average(Ystd**2, weights = ratio_is[:,kk]) for kk in range(ratio_is.shape[1])]))
        return mean, std                                         
    else:  
        psi_y = np.array([psi(Ysimu[kk]) for kk in range(len(Ysimu))])
        mean = np.mean(np.array([np.average(psi_y, weights = ratio_is[:,kk]) for kk in range(ratio_is.shape[1])]))
        return mean
