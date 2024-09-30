import numpy as np
import pandas as pd
from utils_calib import * 
from gp_simus import *
from pymcmcstat.MCMC import MCMC
from pymcmcstat.ParallelMCMC import ParallelMCMC, load_parallel_simulation_results
from pymcmcstat.chain import ChainStatistics as CS
from pymcmcstat.chain import ChainProcessing as CP


# The function plot_transpo computes the prediction mean and standard deviation, from the list of GP means and standard deviations obtained for each posterior sample lambda_k

# The function compute_error computes the RMSRE and the levels associated with the prediction intervals

# The function MCMC_lambda generates MCMC samples $(\boldsymbol{\lambda})_{k=1}^{M}$, taking as argument 
# - "index_calib" the index $t$ of the observed variable $y_t$, 
# - "model_error" a boolean indicating if the parameters representing the model error must be included or not
# - "scale" the standard deviation of the truncated normal prior, 
# - alpha_map the estimated maximum a posterior for the hyperparameters $\boldsymbol{\alpha}$, 
# - "idx_loo" the index of the observation $x_j$ that must be removed in the LOO scheme, 
# - "tune_size" the burnin sample size, 
# - "size" the sample size
# - "mm_list" the parameters of the GP metamodel
# - "results_measures" the dataframe of the observations
# - "sigma" the std deviation of the observation noise
# - "rngseed" the random seed
# - "index_lambda_p" the index of the parameters lambda that are treated with uniform prior
# - "index_lambda_q" the index of the parameters lambda that are treated with hierarchical description
# - "bMINlambda" the lower bounds of the parameters lambda
# - "bMAXlambda" the upper bounds of the parameters lambda

# The function MCMC_lambda_multichains generate multichains MCMC samples for each x_j, and then save the results: the posterior samples and the associated outputs

# The function bayes_lambda_results compute the performance metrics from the obtained posterior outputs

def plot_transpo(Ysimu_list, Ystd_list = None):
    Y_mean = pd.DataFrame(np.zeros((10, 3)))
    Y_std = pd.DataFrame(np.zeros((10, 3)))
    for simu in range(1,11): #for each observation point x_j
        for index_predict in range(1,4): #for each output variable y_t
            list_simus = [Ysimu_list[k].iloc[simu-1, index_predict-1] for k in range(len(Ysimu_list))] #get all the output values (list of length M)
            Y_mean.iloc[simu-1, index_predict-1] = np.mean(list_simus) #get prediction mean
            Y_std.iloc[simu-1, index_predict-1] = np.sqrt(np.var(list_simus)) #get prediction std (without considering std deviation of GP)
            if not (Ystd_list is None):  
                list_var = [Ystd_list[k].iloc[simu-1, index_predict-1]**2 for k in range(len(Ystd_list))] 
                Y_std.iloc[simu-1, index_predict-1] = np.sqrt(Y_std.iloc[simu-1, index_predict-1]**2 + np.mean(list_var))  #Add std deviation of GP                                              
    return pd.concat([Y_mean, Y_std])


def compute_error(Ysimu_list, Ystd_list, true_values):
    res = []
    res_intervals = np.empty((10,0))
    for idx in [1,2,3]: #for each output variable y_t
        simus = np.concatenate([[Ysimu_list[k].iloc[:,idx-1].values] for k in range(len(Ysimu_list))]) #get all the output values (dim M x 10)
        stds = np.concatenate([[Ystd_list[k].iloc[:,idx-1].values] for k in range(len(Ystd_list))]) # get all the stds values (dim M x 10)
        pred = np.apply_along_axis(np.mean, 0,simus) #compute the prediction means
        error = dist2(pred,true_values[f"Y{idx}"]) #compute the rmsre
        res.append(error) 
        eta = abs(pred-true_values[f"Y{idx}"]).values
        intervals = np.apply_along_axis(np.mean, 0, norm.cdf((pred+eta-simus)/stds) - norm.cdf((pred-eta-simus)/stds)).reshape(-1,1) #compute the levels of the predictions intervals
        res_intervals = np.concatenate([res_intervals, intervals], axis=1)
    return res, res_intervals

def MCMC_lambda(index_calib, model_error, scale, alpha_map, idx_loo, tune_size, size, mm_list, results_measures, sigma, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda, rngseed = None): 
    
    def ssfun(theta, data): #log likelihood function
        xdata = data.xdata[0]
        ydata = data.ydata[0]
        lambd = transform_Lambda(Lambda = theta, index_lambda_p= index_lambda_p, index_lambda_q =index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda) 
        Ysimu, Ystd = myCODE(lambd, index = [index_calib],  std_bool = True, vectorize = False, idx_loo = idx_loo, mm_list = mm_list) #get the output for all observations points x_j except at idx_loo
        ss = np.prod(norm.pdf(ydata[:,0]-Ysimu.values.flatten(), loc=0, scale=np.sqrt(sigma[index_calib-1]**2 + Ystd.values.flatten()**2))) #compute the likelihood
        return -2*np.log(ss)

    mcstat = MCMC(rngseed=rngseed)

    x = np.array(list(set(range(len(results_measures))) - set([idx_loo]))) 
    y = results_measures.loc[list(set(range(len(results_measures))) - set([idx_loo])),f"Y{index_calib}"].values #all measures points except loo
    mcstat.data.add_data_set(x, y)
    mcstat.simulation_options.define_simulation_options(
        nsimu=int(tune_size+size),
        updatesigma=False, verbosity = 0, waitbar= True)
    mcstat.model_settings.define_model_settings(sos_function=ssfun)
    
    
    for ii in range(len(index_lambda_p)):
        
        mcstat.parameters.add_model_parameter(
            name=str('$lambd_p_{}$'.format(ii + 1)),
            theta0=0.5,
            minimum=0,
            maximum=1
            ) #uniform prior
    
    if(len(index_lambda_q)) > 0:
        if model_error:
            for ii in range(len(index_lambda_q)):
                mcstat.parameters.add_model_parameter(
                    name=str('$lambd_q_{}$'.format(ii + 1)),
                    theta0=0.5,
                    sample = True, 
                    minimum = 0,
                    maximum = 1, 
                    prior_mu=alpha_map[index_lambda_q[ii]],
                    prior_sigma = scale
                    ) #truncated gaussian prior
        else:
            for ii in range(len(index_lambda_q)):
                mcstat.parameters.add_model_parameter(
                    name=str('$lambd_q_{}$'.format(ii + 1)),
                    theta0=0.5,
                    sample = False
                    ) # if no model error, value of numerical parameters fixed at 0.5
    mcstat.run_simulation()
    return mcstat.simulation_results.results


def MCMC_lambda_multichains(index_calib, model_error, scale, num_chain, tune_size, size,  mm_list, results_measures, sigma, index_lambda_p, index_lambda_q, bMINlambda, bMAXlambda, rngseed, pre_path, loo = True):
    np.random.seed(rngseed)
    seeds = np.random.randint(1000, size = num_chain) #get a random seed for each chain
    if not loo: list_idx_loo = [None]
    else: list_idx_loo = range(len(results_measures))
    alpha_map = None
    list_res = []
    list_res_std = []
    for idx_loo in list_idx_loo: #for each observation point x_j
        if (len(index_lambda_q) > 0) & model_error: 
            alpha_df = pd.read_csv(pre_path + f"/calib_{index_calib}/alpha_df.csv", index_col = 0).values
            if idx_loo is None: alpha_map = alpha_df[0]
            else: alpha_map = alpha_df[idx_loo]
        res = [MCMC_lambda(index_calib = index_calib, model_error = model_error, scale = scale, alpha_map = alpha_map, idx_loo = idx_loo, tune_size = tune_size, size = size,  mm_list = mm_list, results_measures = results_measures, sigma = sigma, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda, rngseed = ss) for ss in seeds] #run every MCMC chain

        samples = np.concatenate([res[i]["chain"][tune_size:,] for i in range(len(res))]) #concatenate the MCMC samples without the burnin phase
        if not model_error: samples = np.concatenate([samples, np.array([res[0]['theta'][len(index_lambda_p):]]*len(samples))], axis = 1)
        
        lambd_post = np.apply_along_axis(lambda x:transform_Lambda(Lambda = x, index_lambda_p = index_lambda_p, index_lambda_q = index_lambda_q, bMINlambda = bMINlambda, bMAXlambda = bMAXlambda), 1, samples) #back to physical values 
        
        if idx_loo is None: save_results(pd.DataFrame(lambd_post), f"lambd_post.csv", pre_path = pre_path, calib = index_calib) #save MCMC samples
        else: save_results(pd.DataFrame(lambd_post), f"lambd_post_{idx_loo}.csv", pre_path = pre_path, calib = index_calib) #save MCMC samples
        
        if not idx_loo is None: 
            res, res_std = myCODE(lambd = lambd_post, index = [1,2,3],  std_bool = True, vectorize = True, idx_loo = idx_loo, new_x = True, mm_list = mm_list) #compute the 3 outputs at idx_loo
            list_res.append(res)
            list_res_std.append(res_std) 
    
    if loo: 
        Ysimu_list = [pd.DataFrame(np.concatenate([list_res[ii].iloc[k,:] for ii in range(len(list_res))]).reshape(len(list_res),3)) for k in range(len(list_res[0]))] #concatenate all outputs
        Ystd_list = [pd.DataFrame(np.concatenate([list_res_std[ii].iloc[k,:] for ii in range(len(list_res_std))]).reshape(len(list_res_std),3)) for k in range(len(list_res_std[0]))] #concatenate all standard deviation
        
    else:
        Ysimu_list,Ystd_list = myCODE(lambd_post, index = [1,2,3],  std_bool = True, vectorize = True, idx_loo = None,  mm_list = mm_list) #compute all outputs
    save_results(pd.concat(Ysimu_list), "Ysimu_list.csv", pre_path = pre_path, calib = index_calib) #save outputs
    save_results(pd.concat(Ystd_list), "Ystd_list.csv", pre_path = pre_path, calib = index_calib) #save std deviations 
        
def bayes_lambda_results(index_calib, pre_path, true_values):
    YY = pd.read_csv(pre_path + f"/calib_{index_calib}/Ysimu_list.csv", index_col = 0) #get the output values
    Ysimu_list = np.array_split(YY, len(YY) // 10)# change the format to a list of dataframes: one df for each lambda
    Ystd = pd.read_csv(pre_path + f"/calib_{index_calib}/Ystd_list.csv", index_col = 0) #get the stds values
    Ystd_list = np.array_split(Ystd, len(Ystd) // 10)# change the format to a list of dataframes: one df for each lambda
    plot1 = plot_transpo(Ysimu_list = Ysimu_list, Ystd_list = Ystd_list) #compute prediction mean and standard deviation
    save_results(plot1, "plot_alpha_map_lamdba_bayesian.csv",pre_path = pre_path, calib = index_calib) #save results
    errors, intervals = compute_error(Ysimu_list, Ystd_list, true_values) #compute RMSRE and p^0.9
    save_results(pd.DataFrame(errors), "errors_map.csv", pre_path = pre_path, calib = index_calib) #save results
    save_results(pd.DataFrame(intervals), "intervals_map.csv", pre_path = pre_path, calib = index_calib) #save results