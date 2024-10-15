import numpy as np
import pandas as pd
from utils_calib import * 
from gp_simus import *
from scipy.stats import norm
from joblib import Parallel, delayed
import sys


# The function MCMC_embed generates MCMC_samples for $(\lambda^1,\lambda^2)$, from 
# - "index_calib" the index of the calibration output
# - "idx_loo" the index of the observation $x_j$ that must be removed in the LOO scheme, 
# - "tune_size" the size of the burnin sample, 
# - "size" the sample size,
# - "u" the multivariate uniform sample of \ksi to compute \lambda^1 + |lambda^2 * u
# - "results_measures" the dataframe of the observations
# - "sigma" the std deviation of the observation noise
# - "index_lambda_p" the index of the parameters lambda that are treated with uniform prior
# - "index_lambda_q" the index of the parameters lambda that are treated with hierarchical description
# - "bMINlambda" the lower bounds of the parameters lambda
# - "bMAXlambda" the upper bounds of the parameters lambda
# - "rngseed" the random seed

# The function MCMC_multichains_idxloo generates multichains MCMC samples for a given x_j (indexed by idx_loo) using MCMC_multichains and save them
# The function MCMC_multichains generates multichains for each x_j
# The function compute_error_embed takes all the GP means and standard deviations and returns the RMSRE and the levels of the predictions intervals
# The function plot_transpo takes the GP means and standard deviations and returns the prediction means and standard deviations
# The function MCMC_treat returns for a given x_j all the simulations $f(x_j, \lambda^1_k + \lambda^2_k \xi_r)$ for $1 \leq k\leq M$ and $1 \leq r\leq R$, more precisely all the GP means and standard deviations
# The function results_embed saves for each calibration index the performance metrics, and predictions means and standard deviations.


def MCMC_embed(index_calib, idx_loo, tune_size, size, u, mm_list, results_measures, sigma, index_lambda_p, bMINlambda, bMAXlambda, rngseed):
    sys.modules['scipy.pi'] = np.pi 
    sys.modules['scipy.cos'] = np.cos
    sys.modules['scipy.sin'] = np.sin 
    from pymcmcstat.MCMC import MCMC
    def priorfun(theta, mu, sigma): #logprior function #theta = (lambda^1, lambda^2)
        lambda1 = theta[:len(index_lambda_p)]
        lambda2 = theta[len(index_lambda_p):]
        if (np.all(lambda2>=0)*int(np.all((lambda1-abs(lambda2))>=0)&np.all((lambda1+abs(lambda2))<=1))) == 0: #On veut s'assurer que lambda^1 + lambda^2 \xi reste dans les bornes
            return 10**10
        else:
            return 0
    
    def ssfun(theta, data): #loglikelihood 
        xdata = data.xdata[0]
        ydata = data.ydata[0]
        lambda1 = theta[:len(index_lambda_p)]
        lambda2 = theta[len(index_lambda_p):]
        lambda_tot = lambda1+lambda2*u #u is a sample of \ksi
        lambda_tot = lambda_tot*(bMAXlambda-bMINlambda)+bMINlambda 
        YY, Ystd = myCODE(lambda_tot, index = [index_calib], std_bool = True, vectorize = True, idx_loo = idx_loo, mm_list = mm_list) #compute gp means and std
        YY = pd.concat([pd.DataFrame(YY[ii].iloc[:, 0]) for ii in range(len(YY))], axis=1)
        Ystd = pd.concat([pd.DataFrame(Ystd[ii].iloc[:, 0]) for ii in range(len(Ystd))], axis=1)
        means = np.apply_along_axis(np.mean, 1, YY) #compute the means of the outputs
        stds = np.sqrt(np.apply_along_axis(np.var,1,YY) + np.apply_along_axis(np.mean,1, Ystd**2) + sigma[index_calib-1]**2) #compute the stds of the outputs
        ss = np.prod(norm.pdf(ydata[:,0], loc = means, scale = stds)) #independent normal approximation
        return -2*np.log(ss) 

    mcstat = MCMC(rngseed=rngseed)
    x = np.array(list(set(range(len(results_measures))) - set([idx_loo])))
    y = results_measures.loc[list(set(range(len(results_measures))) - set([idx_loo])),f"Y{index_calib}"].values
    mcstat.data.add_data_set(x, y)
    mcstat.simulation_options.define_simulation_options(
        nsimu=int(size+tune_size),
        updatesigma=False,verbosity = 0, waitbar= False)
    mcstat.model_settings.define_model_settings(sos_function=ssfun, prior_function = priorfun)

    for ii in range(len(index_lambda_p)):
        mcstat.parameters.add_model_parameter(
            name=str('$lambda1_{}$'.format(ii + 1)),
            theta0=0.5
            )
    for ii in range(len(index_lambda_p)):
        mcstat.parameters.add_model_parameter(
            name=str('$lambda2_{}$'.format(ii + 1)),
            theta0=0.2
            )

    mcstat.run_simulation()

    return mcstat.simulation_results.results


def MCMC_multichains_idxloo(index_calib, idx_loo, num_chain, tune_size, size, u, mm_list, results_measures, sigma, index_lambda_p, bMINlambda, bMAXlambda, rngseed, pre_path):
    np.random.seed(rngseed)
    seeds = np.random.randint(1000, size = num_chain) #get seed for each chain
    res = [MCMC_embed(index_calib = index_calib, idx_loo = idx_loo, tune_size = tune_size, size = size, u = u, mm_list= mm_list, results_measures = results_measures, sigma = sigma, index_lambda_p = index_lambda_p, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda, rngseed = ss) for ss in seeds] #run every MCMC chain
    samples_lambd_post = np.concatenate([res[i]["chain"][tune_size:,] for i in range(len(res))]) #concatenate the chains and remove burnin
    if idx_loo is None: save_results(data = pd.DataFrame(samples_lambd_post), file = f"lambd_post.csv", pre_path = pre_path, calib=index_calib)
    else: save_results(data = pd.DataFrame(samples_lambd_post), file = f"lambd_post_{idx_loo}.csv", pre_path = pre_path, calib=index_calib)
    return pd.DataFrame(samples_lambd_post)
    
def MCMC_multichains(index_calib, num_chain, tune_size, size, u, mm_list, results_measures, sigma, index_lambda_p, bMINlambda, bMAXlambda, rngseed, pre_path, loo = True):
    if not loo: list_idx_loo = [None]
    else: list_idx_loo = range(len(results_measures))
    return Parallel(n_jobs=-1)(delayed(lambda idx_loo: MCMC_multichains_idxloo(index_calib = index_calib, idx_loo = idx_loo, num_chain = num_chain, tune_size = tune_size, size = size, sigma = sigma, u = u, mm_list = mm_list, results_measures = results_measures, index_lambda_p = index_lambda_p, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda, rngseed = rngseed, pre_path = pre_path))(idx_loo) for idx_loo in list_idx_loo)
    
def MCMC_treat(index_calib, idx_loo, u, mm_list, index_lambda_p, bMINlambda, bMAXlambda, pre_path, nb_outputs):
    if idx_loo is None: 
        lambda_post = pd.read_csv(pre_path + f"/calib_{index_calib}/lambd_post.csv", index_col = 0).values #sample of (lambda^1, lambda^2
        Ysimu, Ystd = [], []
    else:  
        lambda_post = pd.read_csv(pre_path + f"/calib_{index_calib}/lambd_post_{idx_loo}.csv", index_col = 0).values
        Ysimu, Ystd = pd.DataFrame(np.zeros((0,nb_outputs))), pd.DataFrame(np.zeros((0,nb_outputs)))
    for i in range(len(lambda_post)):
        lambda1 = lambda_post[i,:len(index_lambda_p)]
        lambda2 = lambda_post[i,len(index_lambda_p):]
        lambda_tot = lambda1 + lambda2*u #u is the sample of ksi
        lambda_tot = np.apply_along_axis(lambda x: x*(bMAXlambda - bMINlambda)+bMINlambda, 1, lambda_tot) #compute GP means and stds
        res, res_std = myCODE(lambda_tot, index = [kk for kk in range(1,nb_outputs+1)],  std_bool = True, vectorize = True, idx_loo = idx_loo, new_x = (idx_loo is not None), mm_list = mm_list) #compute GP means and stds
        if idx_loo is None: 
            Ysimu += res
            Ystd += res_std
        else:
            Ysimu = pd.concat([Ysimu, res], axis = 0) 
            Ystd = pd.concat([Ystd, res_std], axis = 0)
    return Ysimu, Ystd #returns all gp means and stds

def compute_error_embed(simus, stds, true_values):
    res = []
    res_intervals = np.empty((10,0))
    for idx in (np.array(range(len(simus)))+1):
        pred = np.apply_along_axis(np.mean, 0,simus[idx-1]) #get mean prediction
        error = dist2(pred,true_values[f"Y{idx}"])
        res.append(error)
        eta = abs(pred-true_values[f"Y{idx}"]).values
        intervals = np.apply_along_axis(np.mean, 0, norm.cdf((pred+eta-simus[idx-1])/stds[idx-1]) - norm.cdf((pred-eta-simus[idx-1])/stds[idx-1])).reshape(-1,1) #get levels of interval prediction
        res_intervals = np.concatenate([res_intervals, intervals], axis=1)
    return res, res_intervals

def plot_transpo(simus, stds):
    res_pred = pd.DataFrame()
    res_var = pd.DataFrame()
    for idx in (np.array(range(len(simus)))+1): 
        res_pred = pd.concat([res_pred,pd.DataFrame(np.apply_along_axis(np.mean, 0,simus[idx-1]))], axis=1) #get prediction mean
        res_var = pd.concat([res_var,pd.DataFrame(np.apply_along_axis(np.var, 0,simus[idx-1]) + np.apply_along_axis(np.mean, 0,stds[idx-1]**2) )], axis=1) #get prediction variance
    return res_pred, np.sqrt(res_var)


def results_embed(index_calib, num_chain, tune_size, size, u, mm_list, results_measures, sigma, index_lambda_p, bMINlambda, bMAXlambda, pre_path, true_values, rngseed, nb_outputs, loo = True):
    if not loo: 
        results = MCMC_treat(index_calib = index_calib, idx_loo = None, u = u, mm_list = mm_list,index_lambda_p = index_lambda_p, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda, nb_outputs = nb_outputs, pre_path = pre_path)
        YYtot = [pd.concat([results[0][ii].iloc[:,idx_y] for ii in range(len(results[0]))], axis=1).transpose() for idx_y in range(nb_outputs)]
        YYstdtot = [pd.concat([results[1][ii].iloc[:,idx_y] for ii in range(len(results[1]))], axis=1).transpose() for idx_y in range(nb_outputs)]
    else: 
        list_idx_loo = range(len(results_measures))
        results = Parallel(n_jobs=-1)(delayed(lambda idx_loo: MCMC_treat(index_calib = index_calib, idx_loo = idx_loo, u = u, mm_list = mm_list,index_lambda_p = index_lambda_p, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda, nb_outputs = nb_outputs, pre_path = pre_path))(idx_loo) for idx_loo in list_idx_loo)
        YYtot = [pd.concat([results[jj][0].iloc[:,idx_y] for jj in list_idx_loo], axis=1) for idx_y in range(nb_outputs)]
        YYstdtot = [pd.concat([results[jj][1].iloc[:,idx_y] for jj in list_idx_loo], axis=1) for idx_y in range(nb_outputs)]

    errors = compute_error_embed(YYtot, YYstdtot, true_values)
    save_results(data = pd.DataFrame(errors[0]), file = "error_pred.csv", pre_path = pre_path, calib=index_calib)
    save_results(data = pd.DataFrame(errors[1]), file = "interv_errors.csv", pre_path = pre_path, calib=index_calib)

    to_plot = plot_transpo(YYtot, YYstdtot)
    save_results(data = to_plot[0], file = "predictions.csv", pre_path = pre_path, calib=index_calib)
    save_results(data = to_plot[1], file = "std_dev.csv", pre_path = pre_path, calib=index_calib)
    
